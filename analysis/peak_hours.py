"""Phase 3 — where do the models fail? Error by hour of day, peak vs off-peak,
congested vs free-flow, sudden changes, and an example peak-hour forecast.

    python analysis/peak_hours.py --dataset losloop
"""
import argparse
from collections import defaultdict
import numpy as np
import pandas as pd
from common import FIG, TAB, run_dirs, load_pred, bundle_for, df_to_markdown
from tgcn import plotstyle as ps
import matplotlib.pyplot as plt

MODELS = ['naive', 'linear', 'gru', 'gcn', 'tgcn', 'tgcn_selfsep', 'gru_res_time', 'tgcn_res_time']
CONGESTED = {'losloop': 40.0, 'sztaxi': 10.0}   # speed threshold (dataset unit)


def gather(ds):
    by = defaultdict(list)
    for r in run_dirs(ds):
        if r['name'] in MODELS and r['group'] in ('baselines', 'main', 'improve'):
            by[r['name']].append(r['dir'])
    return by


def target_times(b, t_idx, step):
    return b.timestamps[t_idx + step]


def is_peak(ts):
    h = ts.hour + ts.minute / 60
    wk = ts.dayofweek < 5
    return wk & (((h >= 7) & (h < 9)) | ((h >= 16) & (h < 19)))


def main():
    p = argparse.ArgumentParser(); p.add_argument('--dataset', default='losloop')
    ds = p.parse_args().dataset
    b = bundle_for(ds)
    by = gather(ds)
    iv = b.interval_min
    steps = {'15 phút': 15 // iv - 1, '30 phút': 30 // iv - 1, '60 phút': 60 // iv - 1}
    mask_zero = ds == 'sztaxi'
    rows = []; hourly = {}
    for name in MODELS:
        if name not in by: continue
        preds = [load_pred(d) for d in by[name]]
        yt, _, t = preds[0]
        for hname, k in steps.items():
            ts = target_times(b, t, k)
            peak = np.asarray(is_peak(ts))
            y = yt[:, k, :]
            errs = np.stack([pp[1][:, k, :] - y for pp in preds])          # [S,samples,N]
            valid = (y != 0) if mask_zero else np.ones_like(y, bool)
            last = b.values[t - 1]                                         # x_t
            change = np.abs(y - last)
            cong = y < CONGESTED[ds]
            for sel_name, sel in (('cao điểm', peak[:, None] & valid), ('ngoài cao điểm', ~peak[:, None] & valid),
                                  ('kẹt xe (v<%g)' % CONGESTED[ds], cong & valid),
                                  ('thông thoáng', ~cong & valid),
                                  ('biến động mạnh (Δ>10)', (change > 10) & valid),
                                  ('tất cả', valid)):
                e = errs[:, sel]
                rows.append({'model': name, 'horizon': hname, 'subset': sel_name,
                             'share_%': 100 * sel.mean(),
                             'RMSE': float(np.sqrt((e ** 2).mean(1)).mean()),
                             'MAE': float(np.abs(e).mean(1).mean())})
            hr = np.asarray(ts.hour)
            ae = np.abs(errs).mean(0)                                       # [samples,N]
            hourly[(name, hname)] = pd.Series([np.mean(ae[hr == h][valid[hr == h]]) if (hr == h).any() else np.nan
                                               for h in range(24)])
    df = pd.DataFrame(rows)
    df.to_csv(TAB / f'peak_subsets_{ds}.csv', index=False)
    piv = df[df.horizon == '15 phút'].pivot_table(index='model', columns='subset', values='MAE').reindex(
        [m for m in MODELS if m in by])
    piv60 = df[df.horizon == '60 phút'].pivot_table(index='model', columns='subset', values='MAE').reindex(
        [m for m in MODELS if m in by])
    print('MAE 15 min\n', piv.round(3).to_string()); print('MAE 60 min\n', piv60.round(3).to_string())
    (TAB / f'peak_subsets_{ds}.md').write_text(
        '**MAE dự báo 15 phút**\n\n' + df_to_markdown(piv.round(3)) +
        '\n\n**MAE dự báo 60 phút**\n\n' + df_to_markdown(piv60.round(3)), encoding='utf-8')
    share = df[(df.model == MODELS[0]) & (df.horizon == '15 phút')][['subset', 'share_%']]
    print(share.to_string(index=False))

    # MAE by hour of day
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, hname in zip(axes, ('15 phút', '60 phút')):
        for i, name in enumerate(['naive', 'gru', 'tgcn', 'tgcn_res_time']):
            if (name, hname) not in hourly: continue
            s = hourly[(name, hname)]
            ax.plot(s.index, s.values, color=ps.color(name, i), label=ps.label(name),
                    ls='--' if name == 'naive' else '-', marker='o', ms=3.5)
        for a0, a1 in ((7, 9), (16, 19)):
            ax.axvspan(a0, a1, color=ps.GRID, alpha=.6, lw=0)
        ax.set(title=f'Dự báo trước {hname}', xlabel='Giờ trong ngày (thời điểm cần dự báo)',
               xticks=range(0, 24, 3))
    axes[0].set_ylabel('MAE trên tập test'); axes[1].legend(loc='upper right')
    fig.suptitle(f'{ds}: sai số tăng mạnh vào giờ cao điểm (vùng xám)', fontsize=11.5)
    fig.tight_layout(); fig.savefig(FIG / f'peak_mae_by_hour_{ds}.png'); plt.close(fig)

    # example: sensor with the largest speed drop in a test peak
    if 'tgcn_res_time' in by and 'tgcn' in by:
        yt, _, t = load_pred(by['tgcn'][0])
        k = steps['15 phút']
        ts = target_times(b, t, k)
        y = yt[:, k, :]
        peak = np.asarray(is_peak(ts))
        drop = (y[~peak].mean(0) - y[peak].mean(0)) if peak.any() else y.std(0)
        if mask_zero: drop = np.where((y == 0).mean(0) > .05, -np.inf, drop)
        s = int(np.argmax(drop))
        fig, ax = plt.subplots(figsize=(11, 3.8))
        ax.plot(ts, y[:, s], color=ps.INK, lw=2.2, label='Thực tế')
        for i, name in enumerate(['naive', 'gru', 'tgcn', 'tgcn_res_time']):
            if name not in by: continue
            pr = load_pred(by[name][0])[1][:, k, s]
            ax.plot(ts, pr, color=ps.color(name, i), lw=1.4, ls='--' if name == 'naive' else '-',
                    label=ps.label(name), alpha=.95)
        for a0, a1 in ((7, 9), (16, 19)):
            for day in pd.unique(ts.normalize()):
                d = pd.Timestamp(day)
                if d.dayofweek < 5:
                    ax.axvspan(d + pd.Timedelta(hours=a0), d + pd.Timedelta(hours=a1), color=ps.GRID, alpha=.6, lw=0)
        ax.set_xlim(ts[0], ts[-1])
        ax.set(title=f'{ds}: dự báo trước 15 phút tại cảm biến {b.sensor_ids[s]} (tập test, seed 42)',
               ylabel='Tốc độ')
        ax.legend(loc='lower left', ncol=5, fontsize=8)
        fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(FIG / f'peak_example_{ds}.png'); plt.close(fig)
        print('example sensor', b.sensor_ids[s])


if __name__ == '__main__':
    main()
