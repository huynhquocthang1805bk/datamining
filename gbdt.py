"""Gradient-boosted trees (LightGBM) on engineered spatio-temporal features.

A classic data-mining approach to the same task and the same protocol (70/10/20
chronological split, train-only statistics, identical test windows), so its
numbers are directly comparable with GRU / T-GCN.

For every (forecast time t, sensor i) one row with ~45 features:
  own history  : z-scored speed at lags 1..12, differences, rolling mean/std/min/max
  neighbours   : mean of 1-hop neighbours (P z) at lags 1..4, its change, 2-hop mean,
                 slowest neighbour, own-minus-neighbour gap
  calendar     : time of day (fraction, sin, cos), day of week, weekend flag
  profile      : historical mean for the current time-of-day slot (train days) and the
                 profile change expected between now and t+k  (horizon specific)
  sensor       : train mean / std of the sensor, graph degree
Target: change z(t+k) - z(t) (residual formulation), one model per horizon k,
early stopping on the validation split.

    python gbdt.py --dataset losloop --seeds 42 123 2026
    python gbdt.py --dataset sztaxi  --seeds 42
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from tgcn.data import load_bundle, TrafficWindows
from tgcn.metrics import horizon_metrics, metrics
from tgcn.utility import ROOT, save_json

DEFAULT_H = {'losloop': 12, 'sztaxi': 4, 'metrla': 12, 'pemsbay': 12}


def neighbour_ops(bundle):
    a = bundle.adj.astype(np.float64).copy()
    np.fill_diagonal(a, 0.0)
    d = a.sum(1, keepdims=True)
    p = np.divide(a, d, out=np.zeros_like(a), where=d > 0)        # row-normalised, no self
    nb = [np.where(a[i] > 0)[0] for i in range(len(a))]
    return p, nb


def tod_profile(bundle, slots, smooth=2):
    """Per-sensor mean z-score of each time-of-day slot over the TRAIN part (circularly smoothed)."""
    ts = bundle.timestamps
    slot = ((ts.hour * 60 + ts.minute) // bundle.interval_min).to_numpy()
    tr = bundle.splits['train'][1]
    z = bundle.scaled[:tr]
    prof = np.zeros((slots, z.shape[1]))
    cnt = np.zeros(slots)
    for k in range(slots):
        sel = slot[:tr] == k
        if sel.any():
            prof[k] = z[sel].mean(0); cnt[k] = sel.sum()
    kern = np.ones(2 * smooth + 1) / (2 * smooth + 1)
    prof = np.stack([np.convolve(np.r_[prof[-smooth:, j], prof[:, j], prof[:smooth, j]], kern, 'valid')
                     for j in range(prof.shape[1])], axis=1)
    return prof, slot


def build_features(bundle, starts, L, H, p, nb, prof, slot, version=1):
    z = bundle.scaled.astype(np.float64)                          # [T,N]
    T, N = z.shape
    pz = z @ p.T                                                  # 1-hop neighbour mean
    p2z = pz @ p.T                                                # 2-hop
    zmin_nb = np.full((T, N), np.nan)
    for i in range(N):
        if len(nb[i]):
            zmin_nb[:, i] = z[:, nb[i]].min(1)
    s = np.asarray(starts)
    last = s - 1
    lag = np.stack([z[last - j] for j in range(L)], axis=-1)       # [S,N,L] lag1..L
    feats = {f'lag{j + 1}': lag[..., j] for j in range(L)}
    for j in (1, 3, 6, L - 1):
        feats[f'diff_1_{j + 1}'] = lag[..., 0] - lag[..., j]
    feats.update({'mean3': lag[..., :3].mean(-1), 'mean12': lag.mean(-1), 'std12': lag.std(-1),
                  'min12': lag.min(-1), 'max12': lag.max(-1)})
    for j in range(4):
        feats[f'nb_lag{j + 1}'] = pz[last - j]
    feats['nb_diff'] = pz[last] - pz[last - 1]
    feats['nb_diff3'] = pz[last] - pz[last - 3]
    feats['nb2_lag1'] = p2z[last]
    feats['nb_min_lag1'] = zmin_nb[last]
    feats['self_minus_nb'] = z[last] - pz[last]
    ts = bundle.timestamps
    tod = ((ts.hour * 60 + ts.minute) / 1440.0).to_numpy()
    dow = ts.dayofweek.to_numpy()
    bc = lambda v: np.repeat(v[:, None], N, axis=1)
    feats.update({'tod': bc(tod[last]), 'tod_sin': bc(np.sin(2 * np.pi * tod[last])),
                  'tod_cos': bc(np.cos(2 * np.pi * tod[last])), 'dow': bc(dow[last]),
                  'weekend': bc((dow[last] >= 5).astype(float))})
    slots = prof.shape[0]
    feats['prof_now'] = prof[slot[last]]
    feats['dev_from_prof'] = z[last] - prof[slot[last]]
    feats.update({'sensor_mean': np.broadcast_to(bundle.means, (len(s), N)),
                  'sensor_std': np.broadcast_to(bundle.stds, (len(s), N)),
                  'degree': np.broadcast_to(np.array([len(x) for x in nb], float), (len(s), N))})
    extra_k = {}
    if version >= 2:
        # anomaly = deviation from the sensor's own time-of-day profile
        dev = z - prof[slot]                                           # [T,N]
        nb_dev = dev @ p.T
        for j in (1, 3, 6):
            feats[f'dev_lag{j + 1}'] = dev[last - j]
        feats['dev_trend3'] = dev[last] - dev[last - 3]
        feats['nb_dev_lag1'] = nb_dev[last]                            # neighbours abnormally slow?
        feats['nb_dev_trend'] = nb_dev[last] - nb_dev[last - 2]
        net = dev.mean(1); share_slow = (dev < -1.0).mean(1)          # network-wide state
        feats['net_dev'] = bc(net[last]); feats['net_dev_trend'] = bc(net[last] - net[last - 3])
        feats['net_share_slow'] = bc(share_slow[last])
        nb_prof = prof @ p.T                                           # neighbours' profile
        extra_k = {k: (nb_prof[(slot[last] + k) % slots] - nb_prof[slot[last]]).ravel().astype(np.float32)
                   for k in range(1, H + 1)}
    base = pd.DataFrame({k: np.asarray(v, dtype=np.float32).ravel() for k, v in feats.items()})
    prof_change = {k: (prof[(slot[last] + k) % slots] - prof[slot[last]]).ravel().astype(np.float32)
                   for k in range(1, H + 1)}
    return base, prof_change, z[last], extra_k


def run(dataset, seed, L=12, H=None, threads=2, out_root=None, max_train_rows=None, version=1,
        params_override=None, tag='lgbm'):
    H = H or DEFAULT_H[dataset]
    b = load_bundle(ROOT / 'data' / dataset, dataset, L, H)
    p, nb = neighbour_ops(b)
    slots = int(round(1440 / b.interval_min))
    prof, slot = tod_profile(b, slots)
    sets = {}
    for split in ('train', 'val', 'test'):
        st = TrafficWindows(b, split, L, H).starts
        X, pc, zlast, ek = build_features(b, st, L, H, p, nb, prof, slot, version)
        Yz = np.stack([b.scaled[st + k] for k in range(H)], axis=1)          # [S,H,N]
        sets[split] = (st, X, pc, zlast, Yz, ek)
    params = dict(objective='l2', learning_rate=0.05, num_leaves=63, min_child_samples=100,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                  n_estimators=3000, random_state=seed, n_jobs=threads, verbose=-1)
    params.update(params_override or {})
    preds = {sp: np.zeros_like(sets[sp][4]) for sp in ('val', 'test')}
    boosters = []
    importance = []; best_iters = []
    t0 = time.time()
    for k in range(H):
        def xy(sp):
            st, X, pc, zlast, Yz, ek = sets[sp]
            Xk = X.copy(); Xk['prof_change_k'] = pc[k + 1]
            if ek:
                Xk['nb_prof_change_k'] = ek[k + 1]
            yk = (Yz[:, k, :] - zlast).ravel()
            return Xk, yk
        Xtr, ytr = xy('train'); Xva, yva = xy('val')
        if max_train_rows and len(Xtr) > max_train_rows:      # big datasets (METR-LA, PEMS-BAY)
            idx = np.random.default_rng(seed + k).choice(len(Xtr), max_train_rows, replace=False)
            Xtr, ytr = Xtr.iloc[idx], ytr[idx]
        m = lgb.LGBMRegressor(**params)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(100, verbose=False)])
        best_iters.append(int(m.best_iteration_))
        boosters.append(m.booster_)
        importance.append(pd.Series(m.booster_.feature_importance('gain'), index=Xtr.columns, name=k + 1))
        for sp in ('val', 'test'):
            Xs, _ = xy(sp)
            st, _, _, zlast, _, _ = sets[sp]
            preds[sp][:, k, :] = (m.predict(Xs, num_iteration=m.best_iteration_).reshape(len(st), -1) + zlast)
        print(f'  horizon {k + 1}/{H}: best_iter {best_iters[-1]}', flush=True)
    raw = lambda zz: zz * b.stds[None, None, :] + b.means[None, None, :]
    yv, yt = raw(sets['val'][4]), raw(sets['test'][4])
    pv, pt = raw(preds['val']), raw(preds['test'])
    sc = horizon_metrics(yt, pt, b.interval_min)
    if (yt == 0).any():
        sc['masked_nonzero'] = horizon_metrics(yt, pt, b.interval_min, mask_zero=True)
    val_rmse = metrics(yv, pv, mask_zero=False)['RMSE']
    out = Path(out_root or ROOT / 'results' / dataset / 'modern' / tag / f'seed_{seed}')
    out.mkdir(parents=True, exist_ok=True)
    sc.update({'dataset': dataset, 'model': tag, 'seed': seed, 'best_epoch': None, 'caution': '',
               'n_samples': int(len(sets['test'][0])), 'data_is_synthetic': False})
    save_json(sc, out / 'metrics.json')
    save_json({'model': tag, 'feature_version': version, 'params': params, 'best_iterations': best_iters, 'seq_len': L, 'horizon': H,
               'features': list(sets['train'][1].columns) + ['prof_change_k'] + (['nb_prof_change_k'] if version >= 2 else [])},
              out / 'config.json')
    save_json({'best_epoch': None, 'val_rmse_raw': val_rmse, 'train_seconds': time.time() - t0,
               'n_parameters': None}, out / 'training_summary.json')
    pd.DataFrame({'epoch': [1], 'val_mse_scaled': [np.nan], 'val_rmse_raw': [val_rmse]}).to_csv(out / 'history.csv', index=False)
    np.savez_compressed(out / 'test_predictions.npz', y_true=yt.astype(np.float32), y_pred=pt.astype(np.float32),
                        time_indices=sets['test'][0], sensor_ids=np.array(b.sensor_ids))
    np.savez_compressed(out / 'val_predictions.npz', y_true=yv.astype(np.float32), y_pred=pv.astype(np.float32),
                        time_indices=sets['val'][0])          # used by analysis/blend.py (weights fit on val)
    # self-contained model: one booster per horizon + everything needed to build features
    mdir = out / 'model'; mdir.mkdir(exist_ok=True)
    for k, bst in enumerate(boosters):
        bst.save_model(str(mdir / f'h{k + 1:02d}.txt'), num_iteration=best_iters[k])
    np.savez_compressed(mdir / 'stats.npz', means=b.means, stds=b.stds, adj=b.adj, profile=prof,
                        sensor_ids=np.array(b.sensor_ids), interval_min=b.interval_min, seq_len=L, horizon=H,
                        feature_version=version, start_time=str(b.timestamps[0]))
    imp = pd.concat(importance, axis=1).fillna(0)
    imp.to_csv(out / 'feature_importance_gain.csv')
    print(f'{dataset} seed {seed}: test RMSE {sc["overall"]["RMSE"]:.3f} MAE {sc["overall"]["MAE"]:.3f} '
          f'| val RMSE {val_rmse:.3f} | {time.time() - t0:.0f}s', flush=True)
    return sc


class GBDTForecaster:
    """Load a saved LightGBM model directory (results/<ds>/modern/<tag>/seed_<s>/model)
    and forecast the next H steps for every sensor from a speed CSV."""

    def __init__(self, model_dir):
        from types import SimpleNamespace
        mdir = Path(model_dir)
        z = np.load(mdir / 'stats.npz', allow_pickle=False)
        self.st = {k: z[k] for k in z.files}
        self.H = int(self.st['horizon']); self.L = int(self.st['seq_len'])
        self.version = int(self.st['feature_version'])
        def load(k):
            f = mdir / f'h{k + 1:02d}.txt'
            if f.is_file():
                return lgb.Booster(model_file=str(f))
            import gzip                                     # models/ keeps gzip-compressed copies
            with gzip.open(str(f) + '.gz', 'rt') as fh:
                return lgb.Booster(model_str=fh.read())
        self.boosters = [load(k) for k in range(self.H)]
        self.sensor_ids = [str(x) for x in self.st['sensor_ids']]
        self.interval = int(self.st['interval_min'])
        self._ns = SimpleNamespace

    def predict(self, df, end_index, start_time=None):
        """df: speed table (rows = steps, columns = sensor ids); end_index: last observed row."""
        values = df[self.sensor_ids].to_numpy(np.float32)
        t0 = start_time or str(self.st['start_time'])
        b = self._ns(values=values, scaled=((values - self.st['means']) / self.st['stds']).astype(np.float32),
                     means=self.st['means'], stds=self.st['stds'], adj=self.st['adj'], interval_min=self.interval,
                     timestamps=pd.date_range(t0, periods=len(values), freq=f'{self.interval}min'))
        p, nb = neighbour_ops(b)
        prof = self.st['profile']
        slot = ((b.timestamps.hour * 60 + b.timestamps.minute) // self.interval).to_numpy()
        X, pc, zlast, ek = build_features(b, np.array([end_index + 1]), self.L, self.H, p, nb, prof, slot, self.version)
        out = np.zeros((self.H, len(self.sensor_ids)))
        for k, bst in enumerate(self.boosters):
            Xk = X.copy(); Xk['prof_change_k'] = pc[k + 1]
            if ek:
                Xk['nb_prof_change_k'] = ek[k + 1]
            out[k] = bst.predict(Xk) + zlast[0]
        return np.clip(out * self.st['stds'] + self.st['means'], 0, None), b.timestamps


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', default='losloop', choices=list(DEFAULT_H))
    ap.add_argument('--seeds', nargs='+', type=int, default=[42])
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--features', type=int, default=1, choices=[1, 2],
                    help='2 = + neighbour / network-wide anomaly features (results in modern/lgbm_v2)')
    ap.add_argument('--num-leaves', type=int, default=None)
    ap.add_argument('--lr', type=float, default=None)
    ap.add_argument('--tag', default=None, help='output folder name under results/<ds>/modern/')
    ap.add_argument('--max-train-rows', type=int, default=None,
                    help='subsample training rows per horizon (e.g. 2000000 for METR-LA/PEMS-BAY)')
    a = ap.parse_args()
    over = {k: v for k, v in (('num_leaves', a.num_leaves), ('learning_rate', a.lr)) if v is not None}
    tag = a.tag or ('lgbm' if a.features == 1 and not over else
                    'lgbm_v2' if not over else f'lgbm_v{a.features}_' + '_'.join(f'{k}{v}' for k, v in over.items()))
    for s in a.seeds:
        run(a.dataset, s, threads=a.threads, max_train_rows=a.max_train_rows, version=a.features,
            params_override=over, tag=tag)


if __name__ == '__main__':
    main()
