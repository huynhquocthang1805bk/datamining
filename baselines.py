"""Classical baselines evaluated on exactly the same test windows as the networks.

  naive     : persistence, y_{t+k} = x_t (last observed speed). Not in the paper,
              but the minimum bar any forecaster must beat.
  ha_window : mean of the L input steps (what the authors call HA; it is
              horizon-independent, which is why HA has identical numbers for
              15..60 min in Table 1 of the paper).
  ha_tod    : seasonal historical average: mean speed of the same time-of-day
              slot over the training days, per sensor.
  linear    : ridge regression on the L lags (z-scored, shared by all sensors),
              one output per horizon — the linear-kernel SVR of the paper is a
              (regularised) linear model of the same lags.

    python baselines.py --dataset losloop --horizon 12 --seq-len 12
"""
import argparse
import numpy as np
from tgcn.data import load_bundle
from tgcn.metrics import horizon_metrics
from tgcn.utility import ROOT, save_json

DEFAULT_H = {'losloop': 12, 'sztaxi': 4, 'demo': 4, 'metrla': 12, 'pemsbay': 12}


def windows(bundle, split, L, H):
    lo, hi = bundle.splits[split]
    starts = np.arange(lo, hi - H + 1)
    X = np.stack([bundle.values[s - L:s] for s in starts])        # [S,L,N] raw
    Y = np.stack([bundle.values[s:s + H] for s in starts])        # [S,H,N] raw
    return starts, X, Y


def fit_linear(bundle, L, H, alpha=1.0):
    s, X, Y = windows(bundle, 'train', L, H)
    z = lambda a: (a - bundle.means) / bundle.stds
    Xf = z(X).transpose(0, 2, 1).reshape(-1, L)                    # [S*N, L]
    Yf = z(Y).transpose(0, 2, 1).reshape(-1, H)
    Xb = np.hstack([Xf, np.ones((len(Xf), 1))])
    W = np.linalg.solve(Xb.T @ Xb + alpha * np.eye(L + 1), Xb.T @ Yf)
    return W


def predict(name, bundle, X, starts, L, H, W=None):
    S, _, N = X.shape
    if name == 'naive':
        return np.repeat(X[:, -1:, :], H, axis=1)
    if name == 'ha_window':
        return np.repeat(X.mean(1, keepdims=True), H, axis=1)
    if name == 'ha_tod':
        ts = bundle.timestamps
        slot = (ts.hour * 60 + ts.minute).to_numpy() // bundle.interval_min
        tr_end = bundle.splits['train'][1]
        prof = np.zeros((slot.max() + 1, N))
        for k in range(slot.max() + 1):
            sel = np.where(slot[:tr_end] == k)[0]
            prof[k] = bundle.values[sel].mean(0)
        idx = starts[:, None] + np.arange(H)[None, :]
        return prof[slot[idx]]
    if name == 'linear':
        z = (X - bundle.means) / bundle.stds
        Xf = z.transpose(0, 2, 1).reshape(-1, L)
        Pf = np.hstack([Xf, np.ones((len(Xf), 1))]) @ W               # [S*N,H]
        P = Pf.reshape(S, N, H).transpose(0, 2, 1)
        return P * bundle.stds + bundle.means
    raise ValueError(name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='losloop', choices=list(DEFAULT_H))
    p.add_argument('--horizon', type=int, default=None)
    p.add_argument('--seq-len', type=int, default=12)
    a = p.parse_args()
    H = a.horizon or DEFAULT_H[a.dataset]; L = a.seq_len
    bundle = load_bundle(ROOT / 'data' / a.dataset, a.dataset, L, H)
    starts, X, Y = windows(bundle, 'test', L, H)
    W = fit_linear(bundle, L, H)
    for name in ('naive', 'ha_window', 'ha_tod', 'linear'):
        P = predict(name, bundle, X, starts, L, H, W)
        sc = horizon_metrics(Y, P, bundle.interval_min)
        if (Y == 0).any():
            sc['masked_nonzero'] = horizon_metrics(Y, P, bundle.interval_min, mask_zero=True)
        sc.update({'dataset': a.dataset, 'model': name, 'seed': 0, 'best_epoch': None,
                   'data_is_synthetic': a.dataset == 'demo', 'caution': '',
                   'n_samples': int(len(starts))})
        out = ROOT / 'results' / a.dataset / 'baselines' / name
        out.mkdir(parents=True, exist_ok=True)
        save_json(sc, out / 'metrics.json')
        np.savez_compressed(out / 'test_predictions.npz', y_true=Y.astype(np.float32),
                            y_pred=P.astype(np.float32), time_indices=starts,
                            sensor_ids=np.array(bundle.sensor_ids))
        o = sc['overall']; h = sc['per_horizon']
        keys = list(h)
        print(f"{name:10s} overall RMSE {o['RMSE']:.3f} MAE {o['MAE']:.3f} | "
              + ' '.join(f"{k}:{h[k]['RMSE']:.2f}/{h[k]['MAE']:.2f}" for k in keys[::max(1, len(keys)//4)][:4]))


if __name__ == '__main__':
    main()
