"""Phase 1 — data checks and time-series charts (Los-loop, SZ-taxi).

    python analysis/eda.py
Writes reports/figures/eda_*.png and reports/tables/eda_summary.csv
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import shortest_path
from tgcn import plotstyle as ps
from tgcn.data import load_bundle
import matplotlib.pyplot as plt

FIG = ROOT / 'reports' / 'figures'; FIG.mkdir(parents=True, exist_ok=True)
TAB = ROOT / 'reports' / 'tables'; TAB.mkdir(parents=True, exist_ok=True)
NAMES = {'losloop': 'Los-loop (Los Angeles, 5 phút)', 'sztaxi': 'SZ-taxi (Thâm Quyến, 15 phút)'}
PEAKS = [(7, 9), (16, 19)]


def load(ds):
    return load_bundle(ROOT / 'data' / ds, ds, 12, 4 if ds == 'sztaxi' else 12)


def summary(bundles):
    rows = []
    for ds, b in bundles.items():
        v = b.values; raw = pd.read_csv(ROOT / 'data' / ds / 'adjacency.csv', header=None).to_numpy()
        nz = (raw > 0) & ~np.eye(len(raw), dtype=bool)
        tr, va = b.splits['train'][1], b.splits['val'][1]
        rows.append({'dataset': ds, 'T (mốc)': v.shape[0], 'N (nút)': v.shape[1],
                     'bước (phút)': b.interval_min, 'từ': str(b.timestamps[0]),
                     'đến': str(b.timestamps[-1]), 'TB tốc độ': v.mean(), 'độ lệch chuẩn': v.std(),
                     'min': v.min(), 'max': v.max(), '% giá trị 0': 100 * (v == 0).mean(),
                     '% bước không đổi': 100 * (np.diff(v, axis=0) == 0).mean(),
                     'cạnh (không tính self-loop)': int(nz.sum()),
                     'bậc TB': nz.sum(1).mean(), 'kề đối xứng': bool(np.allclose(raw, raw.T)),
                     'train đến': str(b.timestamps[tr]), 'val đến': str(b.timestamps[va])})
    df = pd.DataFrame(rows)
    df.to_csv(TAB / 'eda_summary.csv', index=False)
    print(df.T.to_string())


def daily_profile(bundles):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, (ds, b) in zip(axes, bundles.items()):
        ts = b.timestamps; v = b.values.astype(float)
        if ds == 'sztaxi':
            v = np.where(v == 0, np.nan, v)
        hour = (ts.hour + ts.minute / 60).to_numpy()
        wk = ts.dayofweek.to_numpy() >= 5
        for sel, name, c in ((~wk, 'Ngày thường', ps.SERIES[0]), (wk, 'Cuối tuần', ps.SERIES[1])):
            prof = pd.Series(np.nanmean(v[sel], axis=1)).groupby(hour[sel]).mean()
            ax.plot(prof.index, prof.values, color=c, label=name)
        for a, bb in PEAKS:
            ax.axvspan(a, bb, color=ps.GRID, alpha=.6, lw=0)
        ax.set(title=NAMES[ds], xlabel='Giờ trong ngày', ylabel='Tốc độ TB toàn mạng',
               xticks=range(0, 25, 3))
        ax.legend(loc='lower left')
    axes[0].text(8, axes[0].get_ylim()[1], 'cao điểm sáng', ha='center', va='top', fontsize=8, color=ps.INK2)
    axes[0].text(17.5, axes[0].get_ylim()[1], 'cao điểm chiều', ha='center', va='top', fontsize=8, color=ps.INK2)
    fig.suptitle('Hồ sơ tốc độ theo giờ — kiểm chứng mốc thời gian & giờ cao điểm', fontsize=11.5, color=ps.INK)
    fig.tight_layout(); fig.savefig(FIG / 'eda_daily_profile.png'); plt.close(fig)


def timeline(b, ds, sensors):
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ts = b.timestamps
    for i, s in enumerate(sensors):
        ax.plot(ts, b.values[:, s], lw=1.0, color=ps.SERIES[i], label=f'Cảm biến {b.sensor_ids[s]}')
    tr, va = b.splits['train'][1], b.splits['val'][1]
    ax.axvspan(ts[tr], ts[va - 1], color=ps.SERIES[3], alpha=.10, lw=0)
    ax.axvspan(ts[va], ts[-1], color=ps.SERIES[6], alpha=.08, lw=0)
    y = ax.get_ylim()[1]
    for x, t in ((ts[tr // 2], 'TRAIN 70%'), (ts[(tr + va) // 2], 'VAL 10%'), (ts[(va + len(ts)) // 2], 'TEST 20%')):
        ax.text(x, y, t, ha='center', va='top', fontsize=8.5, color=ps.INK2, weight='bold')
    ax.set(title=f'{NAMES[ds]}: chuỗi tốc độ và cách chia theo thời gian', ylabel='Tốc độ')
    ax.legend(loc='lower left', ncol=3)
    fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(FIG / f'eda_timeline_{ds}.png'); plt.close(fig)


def heatmap(b, ds):
    v = b.values.T.astype(float)
    order = np.argsort(v.mean(1))
    fig, ax = plt.subplots(figsize=(11, 3.8))
    if ds == 'sztaxi':
        v = np.ma.masked_equal(v, 0)
    im = ax.imshow(v[order], aspect='auto', cmap=ps.BLUES.reversed(), interpolation='nearest',
                   extent=[0, v.shape[1] * b.interval_min / 60 / 24, v.shape[0], 0])
    ax.grid(False)
    ax.set(title=f'{NAMES[ds]}: tốc độ theo (cảm biến × thời gian) — màu đậm = chậm/kẹt',
           xlabel='Ngày kể từ mốc đầu', ylabel='Cảm biến (xếp theo tốc độ TB)')
    cb = fig.colorbar(im, ax=ax, pad=.01); cb.outline.set_visible(False)
    fig.tight_layout(); fig.savefig(FIG / f'eda_heatmap_{ds}.png'); plt.close(fig)


def hop_correlation(bundles):
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for ax, (ds, b) in zip(axes, bundles.items()):
        raw = pd.read_csv(ROOT / 'data' / ds / 'adjacency.csv', header=None).to_numpy()
        a = ((raw + raw.T) > 0).astype(float); np.fill_diagonal(a, 0)
        hops = shortest_path(a, unweighted=True)
        tr = b.splits['train'][1]
        v = b.values[:tr].astype(float)
        dv = np.diff(v, axis=0)  # short-term fluctuations (remove daily level)
        ok = v.std(0) > 1e-6
        c_lvl = np.corrcoef(v[:, ok].T); c_d = np.corrcoef(dv[:, ok].T)
        h = hops[np.ix_(ok, ok)]
        iu = np.triu_indices(ok.sum(), 1)
        hh = h[iu]; bins = [(1, '1'), (2, '2'), (3, '3'), (4, '4-5'), (6, '6-9'), (10, '≥10')]
        lab, m1, m2 = [], [], []
        for lo, name in bins:
            hi = {1: 1, 2: 2, 3: 3, 4: 5, 6: 9, 10: np.inf}[lo]
            sel = (hh >= lo) & (hh <= hi) & np.isfinite(hh)
            if sel.sum() == 0: continue
            lab.append(name); m1.append(np.nanmean(c_lvl[iu][sel])); m2.append(np.nanmean(c_d[iu][sel]))
            rows.append({'dataset': ds, 'hop': name, 'n_pairs': int(sel.sum()),
                         'corr_level': m1[-1], 'corr_diff': m2[-1]})
        sel = ~np.isfinite(hh)
        if sel.sum():
            lab.append('không nối'); m1.append(np.nanmean(c_lvl[iu][sel])); m2.append(np.nanmean(c_d[iu][sel]))
            rows.append({'dataset': ds, 'hop': 'disconnected', 'n_pairs': int(sel.sum()),
                         'corr_level': m1[-1], 'corr_diff': m2[-1]})
        x = np.arange(len(lab)); w = .38
        ax.bar(x - w / 2 - .01, m1, w, color=ps.SERIES[0], label='Tương quan mức tốc độ')
        ax.bar(x + w / 2 + .01, m2, w, color=ps.SERIES[1], label='Tương quan biến động Δ(5–15 phút)')
        ax.set(xticks=x, xticklabels=lab, xlabel='Khoảng cách trên đồ thị (số bước nhảy)',
               title=NAMES[ds]); ax.axhline(0, color=ps.NEUTRAL, lw=.8)
    axes[0].set_ylabel('Hệ số tương quan TB (Pearson)')
    axes[0].legend(loc='upper right')
    fig.suptitle('Cảm biến càng gần trên đồ thị thì càng tương quan → cơ sở để dùng GCN', fontsize=11.5)
    fig.tight_layout(); fig.savefig(FIG / 'eda_corr_vs_hop.png'); plt.close(fig)
    pd.DataFrame(rows).to_csv(TAB / 'eda_corr_vs_hop.csv', index=False)


def autocorr(bundles):
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    rows = []
    for i, (ds, b) in enumerate(bundles.items()):
        tr = b.splits['train'][1]; v = b.values[:tr].astype(float)
        v = v[:, v.std(0) > 1e-6]
        lags = np.arange(1, 25 if ds == 'losloop' else 13)
        ac = [np.nanmean([np.corrcoef(v[:-k, j], v[k:, j])[0, 1] for j in range(v.shape[1])]) for k in lags]
        minutes = lags * b.interval_min
        ax.plot(minutes, ac, marker='o', ms=4, color=ps.SERIES[i], label=NAMES[ds])
        rows += [{'dataset': ds, 'lag_min': int(m), 'acf': a} for m, a in zip(minutes, ac)]
    ax.axvline(60, color=ps.NEUTRAL, ls=':', lw=1)
    ax.text(62, .95, '12 bước Los-loop = 60 phút', fontsize=8, color=ps.INK2, va='top')
    ax.set(xlabel='Độ trễ (phút)', ylabel='Tự tương quan TB', ylim=(0, 1),
           title='Tự tương quan theo độ trễ (tập train)')
    ax.legend(loc='upper right'); fig.tight_layout(); fig.savefig(FIG / 'eda_autocorr.png'); plt.close(fig)
    pd.DataFrame(rows).to_csv(TAB / 'eda_autocorr.csv', index=False)


def sz_zeros(b):
    v = b.values; z = v == 0
    ts = b.timestamps; hour = ts.hour.to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    axes[0].hist(100 * z.mean(0), bins=20, color=ps.SERIES[0], rwidth=.92)
    axes[0].set(xlabel='% mốc thời gian có tốc độ = 0', ylabel='Số tuyến đường',
                title='SZ-taxi: tỉ lệ giá trị 0 trên mỗi tuyến')
    zh = pd.Series(z.mean(1)).groupby(hour).mean() * 100
    axes[1].bar(zh.index, zh.values, color=ps.SERIES[1], width=.8)
    axes[1].set(xlabel='Giờ', ylabel='% giá trị 0', title='Tỉ lệ 0 theo giờ (không có taxi ≠ kẹt xe)')
    fig.tight_layout(); fig.savefig(FIG / 'eda_sz_zeros.png'); plt.close(fig)


def main():
    bundles = {ds: load(ds) for ds in ('losloop', 'sztaxi')}
    summary(bundles)
    daily_profile(bundles)
    timeline(bundles['losloop'], 'losloop', [0, 50, 150])
    timeline(bundles['sztaxi'], 'sztaxi', [5, 60])
    heatmap(bundles['losloop'], 'losloop')
    hop_correlation(bundles)
    autocorr(bundles)
    sz_zeros(bundles['sztaxi'])
    print('figures ->', FIG)


if __name__ == '__main__':
    main()
