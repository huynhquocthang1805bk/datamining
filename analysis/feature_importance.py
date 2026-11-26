"""Which inputs does a trained model actually use?

Permutation importance on the TEST set (increase of RMSE when one group of
inputs is shuffled across samples, which breaks its link with the target but
keeps its distribution):
  * lag ℓ          : the speed observed ℓ steps before the forecast time (all sensors)
  * own history    : the 12 past values of the target sensor itself
  * 1-hop neighbours of the target sensor     (spatial information)
  * same number of random non-neighbours      (control: should be ~0)
  * calendar features (time-of-day, day-of-week) for models that use them
Gradient saliency |d yhat_i / d x_j(t-ℓ)| aggregated by graph distance(i,j) and
lag ℓ shows the spatio-temporal receptive field the model has learned.

    python analysis/feature_importance.py --dataset losloop
"""
import argparse
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
from scipy.sparse.csgraph import shortest_path
from common import FIG, TAB, ROOT, run_dirs, load_model, df_to_markdown
from tgcn.data import TrafficWindows
from tgcn import plotstyle as ps
import matplotlib.pyplot as plt

torch.manual_seed(0)


def test_arrays(b, cfg):
    ds = TrafficWindows(b, 'test', cfg['seq_len'], cfg['horizon'])
    L, H = cfg['seq_len'], cfg['horizon']
    X = np.stack([b.scaled[s - L:s] for s in ds.starts]).astype(np.float32)
    TF = np.stack([b.time_features[s - L:s] for s in ds.starts]).astype(np.float32)
    Y = np.stack([b.values[s:s + H] for s in ds.starts]).astype(np.float32)
    return X, TF, Y


@torch.no_grad()
def run(model, X, TF, b):
    out = []
    for i in range(0, len(X), 256):
        out.append(model(torch.from_numpy(X[i:i + 256])[..., None], torch.from_numpy(TF[i:i + 256])).numpy())
    return np.concatenate(out) * b.stds + b.means


def mse_nodes(P, Y, mask_zero):
    e2 = (P - Y) ** 2
    if mask_zero:
        m = Y != 0
        return (e2 * m).sum((0, 1)) / m.sum((0, 1)).clip(1)
    return e2.mean((0, 1))                       # per node [N]


def hops_matrix(b, ds):
    raw = pd.read_csv(ROOT / 'data' / ds / 'adjacency.csv', header=None).to_numpy()
    a = ((raw + raw.T) > 0).astype(float); np.fill_diagonal(a, 0)
    return shortest_path(a, unweighted=True), a


def importance(model, b, cfg, ds, nodes, reps=3):
    mask_zero = ds == 'sztaxi'
    X, TF, Y = test_arrays(b, cfg)
    base = mse_nodes(run(model, X, TF, b), Y, mask_zero)
    rmse0 = np.sqrt(base.mean())
    rng = np.random.default_rng(0)
    rows = []
    L = cfg['seq_len']
    # lags (all sensors at once)
    for lag in range(1, L + 1):
        inc = []
        for _ in range(reps):
            Xp = X.copy(); perm = rng.permutation(len(X))
            Xp[:, L - lag] = X[perm, L - lag]
            inc.append(np.sqrt(mse_nodes(run(model, Xp, TF, b), Y, mask_zero).mean()) - rmse0)
        rows.append({'feature': f'lag {lag} ({lag * b.interval_min} phút trước)', 'group': 'lag',
                     'lag': lag, 'ΔRMSE': float(np.mean(inc))})
    # calendar features
    if cfg.get('time_features') or (cfg['model'] == 'tgcn_plus' and not cfg.get('no_time_features')):
        inc = []
        for _ in range(reps):
            perm = rng.permutation(len(X))
            inc.append(np.sqrt(mse_nodes(run(model, X, TF[perm], b), Y, mask_zero).mean()) - rmse0)
        rows.append({'feature': 'đặc trưng thời gian (giờ, thứ)', 'group': 'time', 'lag': None,
                     'ΔRMSE': float(np.mean(inc))})
    # spatial: per target node, shuffle own / neighbours / random non-neighbours
    hops, a = hops_matrix(b, ds)
    sp = defaultdict(list)
    for i in nodes:
        nb = np.where(a[i] > 0)[0]
        if len(nb) == 0: continue
        far = np.where((hops[i] > 3) | ~np.isfinite(hops[i]))[0]
        rnd = rng.choice(far, size=min(len(nb), len(far)), replace=False)
        for kind, cols in (('lịch sử của chính nút', np.array([i])), ('láng giềng 1-hop', nb),
                           ('nút ngẫu nhiên xa (≥4 hop)', rnd)):
            perm = rng.permutation(len(X))
            Xp = X.copy(); Xp[:, :, cols] = X[perm][:, :, cols]
            m = mse_nodes(run(model, Xp, TF, b), Y, mask_zero)
            sp[kind].append(np.sqrt(m[i]) - np.sqrt(base[i]))
    for kind, v in sp.items():
        rows.append({'feature': kind, 'group': 'spatial', 'lag': None, 'ΔRMSE': float(np.mean(v)),
                     'ΔRMSE_std': float(np.std(v)), 'n_nodes': len(v)})
    return pd.DataFrame(rows), rmse0


def saliency(model, b, cfg, ds, n_samples=48, n_targets=40, step=None):
    X, TF, _ = test_arrays(b, cfg)
    hops, _ = hops_matrix(b, ds)
    rng = np.random.default_rng(1)
    idx = rng.choice(len(X), size=min(n_samples, len(X)), replace=False)
    targets = rng.choice(X.shape[2], size=min(n_targets, X.shape[2]), replace=False)
    k = step if step is not None else min(2, cfg['horizon'] - 1)
    x = torch.from_numpy(X[idx])[..., None].requires_grad_(True)
    tf = torch.from_numpy(TF[idx])
    L = cfg['seq_len']
    bins = [0, 1, 2, 3, 4, 6, np.inf]; names = ['0 (chính nó)', '1', '2', '3', '4-5', '≥6']
    acc = np.zeros((len(names), L)); cnt = np.zeros(len(names))
    model.eval()
    for i in targets:
        if x.grad is not None: x.grad = None
        out = model(x, tf)[:, k, i].sum()
        g = torch.autograd.grad(out, x)[0].abs().mean(0)[..., 0].numpy()   # [L,N]
        h = hops[i]
        for bi in range(len(names)):
            sel = (h >= bins[bi]) & (h < bins[bi + 1]) if bi else (h == 0)
            if sel.any():
                acc[bi] += g[:, sel].mean(1); cnt[bi] += 1
    sal = acc / cnt[:, None].clip(1)
    return pd.DataFrame(sal[:, ::-1], index=names, columns=[f'lag {l}' for l in range(1, L + 1)])


def plot(res, sal, models, interval, ds):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    for i, name in enumerate(models):
        s = res[(res.model == name) & (res.group == 'lag')]
        axes[0].plot(s.lag * interval, s['ΔRMSE'], marker='o', ms=4, color=ps.color(name, i),
                     label=ps.label(name))
    axes[0].set(xlabel='Giá trị quan sát cách thời điểm dự báo (phút)', ylabel='RMSE tăng khi xáo trộn',
                title='Tầm quan trọng của từng độ trễ')
    axes[0].legend(fontsize=7.5)
    kinds = ['lịch sử của chính nút', 'láng giềng 1-hop', 'nút ngẫu nhiên xa (≥4 hop)', 'đặc trưng thời gian (giờ, thứ)']
    x = np.arange(len(kinds)); w = .8 / len(models)
    for i, name in enumerate(models):
        s = res[res.model == name].set_index('feature').reindex(kinds)
        axes[1].bar(x + (i - (len(models) - 1) / 2) * w, s['ΔRMSE'].fillna(0), w * .92,
                    color=ps.color(name, i), label=ps.label(name))
    axes[1].set(xticks=x, xticklabels=['Lịch sử\nchính nút', 'Láng giềng\n1-hop', 'Nút xa\n(đối chứng)', 'Giờ/thứ'],
                ylabel='RMSE tại nút tăng khi xáo trộn', title='Nguồn thông tin: thời gian vs không gian')
    axes[1].axhline(0, color=ps.INK2, lw=.8)
    fig.suptitle(f'{ds}: ảnh hưởng của thuộc tính đầu vào (permutation importance trên test)', fontsize=11.5, y=1.02)
    fig.tight_layout(); fig.savefig(FIG / f'fi_importance_{ds}.png'); plt.close(fig)
    if sal:
        fig, axes = plt.subplots(1, len(sal), figsize=(3.7 * len(sal) + 1.2, 3.8), squeeze=False, sharey=True)
        vmax = max(float(v.values.max()) for v in sal.values())
        short = {'gru': 'GRU', 'tgcn': 'T-GCN (gốc)', 'tgcn_selfsep_res_time': 'T-GCN-RTS (tách self)'}
        for ax, (name, sm) in zip(axes[0], sal.items()):
            im = ax.imshow(sm.values, aspect='auto', cmap=ps.BLUES, vmin=0, vmax=vmax)
            ax.grid(False)
            ax.set(xticks=range(0, sm.shape[1], 2), xticklabels=[f'{(j + 1) * interval}' for j in range(0, sm.shape[1], 2)],
                   yticks=range(len(sm.index)), yticklabels=sm.index, xlabel='Độ trễ (phút trước)',
                   title=short.get(name, name))
        axes[0][0].set_ylabel('Khoảng cách tới nút dự báo (hop)')
        cb = fig.colorbar(im, ax=axes[0].tolist(), pad=.015, fraction=.03); cb.outline.set_visible(False)
        cb.set_label('|∂ŷ/∂x| trung bình')
        fig.suptitle(f'{ds}: trường tiếp nhận không gian–thời gian (gradient saliency, dự báo 15 phút)',
                     fontsize=11.5, y=1.03)
        fig.savefig(FIG / f'fi_saliency_{ds}.png'); plt.close(fig)


def pick(ds, name):
    for r in run_dirs(ds):
        if r['name'] == name and r['seed'] == 42 and r['group'] in ('main', 'improve', 'variants'):
            return r['dir']
    return None


def main():
    p = argparse.ArgumentParser(); p.add_argument('--dataset', default='losloop')
    p.add_argument('--nodes', type=int, default=60)
    p.add_argument('--plot-only', action='store_true', help='redraw figures from saved CSVs')
    a = p.parse_args(); ds = a.dataset
    if a.plot_only:
        res = pd.read_csv(TAB / f'feature_importance_{ds}.csv')
        models = list(dict.fromkeys(res.model))
        sal = {m: pd.read_csv(TAB / f'fi_saliency_{m}_{ds}.csv', index_col=0)
               for m in ('gru', 'tgcn', 'tgcn_selfsep_res_time') if (TAB / f'fi_saliency_{m}_{ds}.csv').is_file()}
        plot(res, sal, models, 5 if ds == 'losloop' else 15, ds)
        return
    models = [m for m in ('gru', 'tgcn', 'gru_res_time', 'tgcn_res_time', 'tgcn_selfsep_res_time') if pick(ds, m)]
    allrows = []; sal = {}
    for name in models:
        model, b, cfg = load_model(pick(ds, name))
        nodes = np.random.default_rng(2).choice(len(b.sensor_ids), size=min(a.nodes, len(b.sensor_ids)),
                                                replace=False)
        df, r0 = importance(model, b, cfg, ds, nodes)
        df['model'] = name; df['RMSE_base'] = r0; allrows.append(df)
        print(name, 'base RMSE', round(r0, 4)); print(df.round(4).to_string())
        if name in ('gru', 'tgcn', 'tgcn_selfsep_res_time'):
            sal[name] = saliency(model, b, cfg, ds)
            sal[name].to_csv(TAB / f'fi_saliency_{name}_{ds}.csv')
    res = pd.concat(allrows); res.to_csv(TAB / f'feature_importance_{ds}.csv', index=False)
    plot(res, sal, models, b.interval_min, ds)
    md = res[res.group != 'lag'][['model', 'feature', 'ΔRMSE']].round(4)
    (TAB / f'feature_importance_{ds}.md').write_text(df_to_markdown(md, index=False), encoding='utf-8')


if __name__ == '__main__':
    main()
