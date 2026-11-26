"""Horizon-wise blending (stacking) fitted on the VALIDATION split only.

For every forecast step k we choose non-negative weights w_k (sum 1, grid 0.05)
over a few members (each = mean of its seeds) that minimise the validation MSE,
then apply them unchanged to the test split. The test set is never used to
choose anything.

    python analysis/blend.py --dataset losloop --members lgbm gru_rtp gru_res_time
"""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd
from common import ROOT, FIG, TAB, run_dirs, load_pred, load_model, bundle_for
from reliability import predict_split
from tgcn.metrics import horizon_metrics
from tgcn.utility import save_json
from tgcn import plotstyle as ps
import matplotlib.pyplot as plt

NAMES = {'lgbm': 'LightGBM', 'lgbm_v2': 'LightGBM v2', 'lgbm_best': 'LightGBM', 'gru_rtp': 'GRU-RTP', 'gru_res_time': 'GRU-RT',
         'tgcn_rtsp': 'T-GCN-RTSP', 'tgcn_selfsep_res_time': 'T-GCN-RTS', 'stid_res': 'STID + res',
         'tgcn_selfsep': 'T-GCN tách self', 'dcrnn': 'DCRNN-lite', 'gwnet': 'Graph WaveNet',
         'gwnet_res': 'Graph WaveNet + res', 'gru': 'GRU', 'stid': 'STID',
         'gpu_gwnet': 'Graph WaveNet (GPU)', 'gpu_gwnet_res': 'Graph WaveNet + res (GPU)',
         'gpu_staeformer': 'STAEformer (GPU)', 'gpu_staeformer_full': 'STAEformer đầy đủ (GPU)'}


def member_runs(ds, name):
    if name.startswith('gpu_'):    # GPU runs copied into results/<ds>/modern_gpu/<name>/seed_*/
        return [r for r in run_dirs(ds) if r['name'] == name[4:] and r['group'] == 'modern_gpu']
    return [r for r in run_dirs(ds) if r['name'] == name and r['group'] in ('main', 'improve', 'variants', 'modern')]


def val_test_preds(ds, name):
    rs = member_runs(ds, name)
    if not rs:
        raise SystemExit(f'no runs for {name}')
    vals, tests = [], []
    for r in rs:
        yt, pt, t = load_pred(r['dir'])
        vp = Path(r['dir']) / 'val_predictions.npz'
        if vp.is_file():
            z = np.load(vp); yv, pv = z['y_true'], z['y_pred']
        else:
            model, b, cfg = load_model(r['dir'])
            yv, pv, _, _ = predict_split(model, b, cfg, 'val')
            np.savez_compressed(vp, y_true=yv.astype(np.float32), y_pred=pv.astype(np.float32))
        vals.append(pv); tests.append(pt)
    return yv, np.mean(vals, 0), yt, np.mean(tests, 0), t, len(rs)


def simplex(m, step=0.05):
    n = int(round(1 / step))
    for c in itertools.product(range(n + 1), repeat=m - 1):
        if sum(c) <= n:
            yield np.array([*c, n - sum(c)]) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='losloop')
    ap.add_argument('--members', nargs='+', default=['lgbm', 'gru_rtp', 'gru_res_time'])
    ap.add_argument('--tag', default='blend')
    a = ap.parse_args(); ds = a.dataset
    mask_zero = ds == 'sztaxi'
    b = bundle_for(ds)
    data = {m: val_test_preds(ds, m) for m in a.members}
    yv = data[a.members[0]][0]; yt = data[a.members[0]][2]; t = data[a.members[0]][4]
    H = yv.shape[1]
    W = np.zeros((H, len(a.members)))
    pt = np.zeros_like(yt, dtype=np.float64)
    for k in range(H):
        vk = np.stack([data[m][1][:, k] for m in a.members])               # [M,S,N]
        tk = np.stack([data[m][3][:, k] for m in a.members])
        mk = (yv[:, k] != 0) if mask_zero else np.ones_like(yv[:, k], bool)
        best, bw = np.inf, None
        for w in simplex(len(a.members)):
            e = np.tensordot(w, vk, 1) - yv[:, k]
            mse = (e[mk] ** 2).mean()
            if mse < best:
                best, bw = mse, w
        W[k] = bw
        pt[:, k] = np.tensordot(bw, tk, 1)
    sc = horizon_metrics(yt, pt, b.interval_min)
    if (yt == 0).any():
        sc['masked_nonzero'] = horizon_metrics(yt, pt, b.interval_min, mask_zero=True)
    out = ROOT / 'results' / ds / 'ensemble' / a.tag
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / 'test_predictions.npz', y_true=yt, y_pred=pt.astype(np.float32), time_indices=t)
    save_json({**sc, 'dataset': ds, 'model': a.tag, 'seed': 0, 'caution': '', 'members': a.members,
               'weights_per_horizon': W.tolist()}, out / 'metrics.json')
    wdf = pd.DataFrame(W, columns=[NAMES.get(m, m) for m in a.members],
                       index=[f'{(k + 1) * b.interval_min} phút' for k in range(H)])
    wdf.to_csv(TAB / f'{a.tag}_weights_{ds}.csv')
    view = sc.get('masked_nonzero', sc) if mask_zero else sc
    print(wdf.round(2).to_string())
    print(f"{a.tag}: test RMSE {view['overall']['RMSE']:.3f} MAE {view['overall']['MAE']:.3f}")
    fig, ax = plt.subplots(figsize=(8, 3.4))
    x = np.arange(H) + 1
    bottom = np.zeros(H)
    for i, col in enumerate(wdf.columns):
        ax.bar(x * b.interval_min, wdf[col], bottom=bottom, width=b.interval_min * 0.8, color=ps.SERIES[[3, 2, 0, 6, 4][i % 5]],
               label=col)
        bottom += wdf[col].values
    ax.set(xlabel='Tầm dự báo (phút)', ylabel='Trọng số (fit trên validation)', ylim=(0, 1),
           title=f'{ds}: trọng số trộn theo từng tầm dự báo')
    ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.01, 1.0)); ax.grid(axis='x', visible=False)
    fig.tight_layout(); fig.savefig(FIG / f'{a.tag}_weights_{ds}.png'); plt.close(fig)


if __name__ == '__main__':
    main()
