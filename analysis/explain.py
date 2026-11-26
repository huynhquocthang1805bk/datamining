"""Explain *why* the results look the way they do.

 1. skill score vs persistence per horizon: SS = 1 - MSE_model / MSE_naive
 2. update gate u of the T-GCN cell vs traffic state (how the GRU decides to keep
    memory or overwrite it with the new, graph-aggregated observation)
 3. residual models: predicted change vs actual change
 4. model card: every learned tensor, its shape and role

    python analysis/explain.py --dataset losloop
"""
import argparse
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
from common import FIG, TAB, run_dirs, load_model, load_pred, df_to_markdown
from tgcn.data import TrafficWindows
from tgcn import plotstyle as ps
import matplotlib.pyplot as plt

ROLE = {
    'cell.gates.linear.weight': 'W_g: GCN trong cổng reset r và update u, [2H, 1(+4)+H]',
    'cell.gates.linear.bias': 'b_g: bias cổng (khởi tạo 1.0 như mã tác giả)',
    'cell.candidate.linear.weight': 'W_c: GCN tạo trạng thái ứng viên c, [H, 1(+4)+H]',
    'cell.candidate.linear.bias': 'b_c',
    'head.weight': 'W_out: ánh xạ trạng thái ẩn → H bước dự báo',
    'head.bias': 'b_out',
    'adj': 'Â = D^-1/2 (A+I) D^-1/2 (cố định, không học)',
}


def by_name(ds):
    by = defaultdict(list)
    for r in run_dirs(ds):
        if r['group'] in ('baselines', 'main', 'improve', 'variants'):
            by[r['name']].append(r)
    return by


def skill(by, ds):
    if 'naive' not in by: return None
    naive = by['naive'][0]['metrics']['per_horizon']
    ks = list(naive)
    key = lambda m: m.get('masked_nonzero', m) if ds == 'sztaxi' else m
    naive = key(by['naive'][0]['metrics'])['per_horizon']
    rows = []
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for i, name in enumerate(['linear', 'gcn', 'gru', 'tgcn', 'tgcn_selfsep', 'gru_res_time', 'tgcn_res_time',
                              'tgcn_selfsep_res_time']):
        if name not in by: continue
        ss = np.array([[1 - key(r['metrics'])['per_horizon'][k]['RMSE'] ** 2 / naive[k]['RMSE'] ** 2 for k in ks]
                       for r in by[name]])
        x = [int(k[:-3]) for k in ks]
        ax.plot(x, 100 * ss.mean(0), marker='o', ms=4, color=ps.color(name, i), label=ps.label(name))
        for k, v in zip(ks, ss.mean(0)):
            rows.append({'model': name, 'horizon': k, 'skill_vs_naive_%': 100 * v})
    ax.axhline(0, color=ps.INK2, lw=1, ls='--')
    lo = min(r['skill_vs_naive_%'] for r in rows)
    if lo < -40:
        ax.set_ylim(-40, None)
        worst = min(rows, key=lambda r: r['skill_vs_naive_%'])
        ax.text(x[0], -38, f"{ps.label(worst['model'])}: {worst['skill_vs_naive_%']:.0f}% ở {worst['horizon']} (ngoài khung)",
                fontsize=8, color=ps.INK2, va='bottom')
    ax.text(x[-1], 1, 'Naive (giữ giá trị cuối) = 0%', ha='right', va='bottom', fontsize=8, color=ps.INK2)
    ax.set(xlabel='Tầm dự báo (phút)', ylabel='Skill = 1 − MSE/MSE_naive (%)',
           title=f'{ds}: mô hình hơn "giữ nguyên giá trị cuối" bao nhiêu?')
    ax.legend(fontsize=8, loc='lower right'); fig.tight_layout()
    fig.savefig(FIG / f'exp_skill_{ds}.png'); plt.close(fig)
    return pd.DataFrame(rows)


@torch.no_grad()
def gates(run_dir, ds):
    model, b, cfg = load_model(run_dir)
    st = TrafficWindows(b, 'test', cfg['seq_len'], cfg['horizon']).starts
    L = cfg['seq_len']
    X = torch.from_numpy(np.stack([b.scaled[s - L:s] for s in st]))[..., None]
    TF = torch.from_numpy(np.stack([b.time_features[s - L:s] for s in st]))
    _, g, h = model(X, TF, return_gates=True)
    u_last = g[-1][1].mean(-1).numpy()                     # [S,N] update gate, last step
    r_last = g[-1][0].mean(-1).numpy()
    raw = np.stack([b.values[s - L:s] for s in st])        # [S,L,N]
    dlast = np.abs(raw[:, -1] - raw[:, -2])                # most recent change
    level = raw[:, -1]
    if ds == 'sztaxi':
        keep = level > 0
    else:
        keep = np.ones_like(level, bool)
    df = pd.DataFrame({'u': u_last[keep], 'r': r_last[keep], 'dlast': dlast[keep], 'level': level[keep]})
    return df


def plot_gates(dfs, ds):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for i, (name, df) in enumerate(dfs.items()):
        q = pd.qcut(df.dlast.rank(method='first'), 10, labels=False)
        g = df.groupby(q).agg(dl=('dlast', 'median'), u=('u', 'mean'))
        axes[0].plot(g.dl, g.u, marker='o', ms=4, color=ps.color(name, i), label=ps.label(name))
        bins = [0, 20, 30, 40, 50, 60, 75] if ds == 'losloop' else [0, 5, 10, 15, 20, 30, 90]
        g2 = df.groupby(pd.cut(df.level, bins), observed=True)['u'].mean()
        axes[1].plot([str(x) for x in g2.index], g2.values, marker='o', ms=4, color=ps.color(name, i),
                     label=ps.label(name))
    axes[0].set(xlabel='|Δ tốc độ| ở bước gần nhất (trung vị mỗi thập phân vị)', ylabel='Cổng cập nhật u (TB)',
                title='Biến động càng lớn → u càng nhỏ (ghi đè bộ nhớ)', xscale='symlog')
    axes[1].set(xlabel='Tốc độ hiện tại', title='u theo trạng thái giao thông')
    axes[0].legend(fontsize=8)
    fig.suptitle(f'{ds}: cổng GRU bên trong T-GCN phản ứng với trạng thái giao thông', fontsize=11.5)
    fig.tight_layout(); fig.savefig(FIG / f'exp_gates_{ds}.png'); plt.close(fig)


def delta_scatter(run_dir, b_interval, ds, name):
    yt, yp, t = load_pred(run_dir)
    from common import bundle_for
    b = bundle_for(ds)
    last = b.values[t - 1]                                 # [S,N]
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, k in zip(axes, (min(2, yt.shape[1] - 1), yt.shape[1] - 1)):
        da = (yt[:, k] - last).ravel(); dp = (yp[:, k] - last).ravel()
        keep = (yt[:, k].ravel() != 0) & (last.ravel() != 0) if ds == 'sztaxi' else np.ones_like(da, bool)
        da, dp = da[keep], dp[keep]
        r2 = 1 - np.sum((dp - da) ** 2) / np.sum((da - da.mean()) ** 2)
        corr = np.corrcoef(da, dp)[0, 1]
        rows.append({'model': name, 'horizon_min': (k + 1) * b_interval, 'R2_of_change': r2,
                     'corr': corr, 'slope': np.polyfit(da, dp, 1)[0]})
        ax.hexbin(da, dp, gridsize=60, bins='log', cmap=ps.BLUES, mincnt=1, extent=(-40, 40, -40, 40))
        ax.plot([-40, 40], [-40, 40], color=ps.INK2, lw=1, ls='--')
        ax.set(xlim=(-40, 40), ylim=(-40, 40), xlabel='Thay đổi thực tế  y(t+k) − x(t)',
               ylabel='Thay đổi dự báo  ŷ(t+k) − x(t)',
               title=f'{(k + 1) * b_interval} phút: corr={corr:.2f}, hệ số góc={rows[-1]["slope"]:.2f}')
        ax.grid(False)
    fig.suptitle(f'{ds} — {ps.label(name)}: mô hình dự báo được bao nhiêu phần "thay đổi"?', fontsize=11.5)
    fig.tight_layout(); fig.savefig(FIG / f'exp_delta_{name}_{ds}.png'); plt.close(fig)
    return rows


def model_card(run_dir):
    model, b, cfg = load_model(run_dir)
    rows = []
    for k, v in list(model.state_dict().items()):
        rows.append({'tensor': k, 'shape': 'x'.join(map(str, v.shape)), 'n': int(v.numel()),
                     'learned': k != 'adj', 'role': ROLE.get(k, '')})
    return pd.DataFrame(rows), cfg


def main():
    p = argparse.ArgumentParser(); p.add_argument('--dataset', default='losloop')
    ds = p.parse_args().dataset
    by = by_name(ds)
    sk = skill(by, ds)
    if sk is not None:
        sk.to_csv(TAB / f'exp_skill_{ds}.csv', index=False)
        print(sk.pivot_table(index='model', columns='horizon', values='skill_vs_naive_%', sort=False).round(1).to_string())
    sel = lambda n: next((r['dir'] for r in by.get(n, []) if r['seed'] == 42), None)
    dfs = {n: gates(sel(n), ds) for n in ('tgcn', 'tgcn_res_time') if sel(n)}
    if dfs:
        plot_gates(dfs, ds)
        for n, d in dfs.items():
            print(n, 'corr(u, |Δ|)=%.3f  corr(u, level)=%.3f' % (d.u.corr(d.dlast, method='spearman'),
                                                                 d.u.corr(d.level, method='spearman')))
    rows = []
    for n in ('tgcn', 'tgcn_res_time', 'gru'):
        if sel(n):
            rows += delta_scatter(sel(n), 5 if ds == 'losloop' else 15, ds, n)
    if rows:
        dd = pd.DataFrame(rows); dd.to_csv(TAB / f'exp_delta_{ds}.csv', index=False); print(dd.round(3).to_string())
    if sel('tgcn_res_time'):
        card, cfg = model_card(sel('tgcn_res_time'))
        card.to_csv(TAB / f'model_card_tgcn_res_time_{ds}.csv', index=False)
        (TAB / f'model_card_tgcn_res_time_{ds}.md').write_text(df_to_markdown(card, index=False), encoding='utf-8')
        print(card.to_string()); print('learned parameters:', card[card.learned].n.sum())


if __name__ == '__main__':
    main()
