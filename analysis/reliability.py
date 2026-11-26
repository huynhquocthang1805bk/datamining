"""How much can the numbers be trusted?

 1. seed variability (mean, std, coefficient of variation)
 2. paired moving-block bootstrap CI of the RMSE difference + Diebold-Mariano test
 3. robustness: Gaussian noise on the inputs and random sensor failures (input = 0)
 4. split-conformal prediction intervals (calibrated on validation) -> test coverage
 5. metrics excluding targets that lie on linearly-interpolated segments (Los-loop)
 6. bias by speed level (does the model over-estimate congested speeds?)

    python analysis/reliability.py --dataset losloop
"""
import argparse
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
from scipy import stats
from common import FIG, TAB, ROOT, run_dirs, load_pred, load_model, bundle_for, df_to_markdown
from tgcn.data import TrafficWindows, inverse_transform
from tgcn.metrics import metrics
from tgcn import plotstyle as ps
import matplotlib.pyplot as plt

RNG = np.random.default_rng(0)


def runs_by_name(ds):
    by = defaultdict(list)
    for r in run_dirs(ds):
        if r['group'] in ('baselines', 'main', 'improve', 'variants', 'ensemble'):
            by[r['name'] if r['group'] != 'ensemble' else 'ensemble_' + r['name']].append(r)
    return by


# ---------- 1. seeds ----------
def seed_table(by):
    rows = []
    for name, rs in by.items():
        if len(rs) < 2: continue
        v = np.array([r['metrics']['overall']['RMSE'] for r in rs])
        m = np.array([r['metrics']['overall']['MAE'] for r in rs])
        ep = [r['metrics'].get('best_epoch') for r in rs]
        rows.append({'model': name, 'seeds': len(rs), 'RMSE_mean': v.mean(), 'RMSE_std': v.std(ddof=1),
                     'RMSE_CV_%': 100 * v.std(ddof=1) / v.mean(), 'MAE_mean': m.mean(),
                     'MAE_std': m.std(ddof=1), 'best_epochs': ep})
    return pd.DataFrame(rows).sort_values('RMSE_mean')


# ---------- 2. bootstrap + DM ----------
def sample_losses(rs, mask_zero):
    """Per test sample squared error averaged over nodes & horizons (and seeds)."""
    out = []
    for r in rs:
        yt, yp, t = load_pred(r['dir'])
        e2 = (yp - yt) ** 2
        if mask_zero:
            m = yt != 0
            out.append((e2 * m).sum((1, 2)) / m.sum((1, 2)).clip(1))
        else:
            out.append(e2.mean((1, 2)))
    return np.mean(out, 0), t


def block_bootstrap(d, block, n_boot=4000):
    n = len(d); nb = int(np.ceil(n / block))
    starts = RNG.integers(0, n - block + 1, size=(n_boot, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :n]
    return d[idx].mean(1)


def dm_test(d, lag):
    """Diebold-Mariano with Newey-West variance (two-sided)."""
    n = len(d); dc = d - d.mean()
    gamma = [np.dot(dc[k:], dc[:n - k]) / n for k in range(lag + 1)]
    var = gamma[0] + 2 * sum((1 - k / (lag + 1)) * gamma[k] for k in range(1, lag + 1))
    dm = d.mean() / np.sqrt(var / n)
    return float(dm), float(2 * (1 - stats.norm.cdf(abs(dm))))


def pairwise(by, pairs, block, mask_zero):
    rows = []
    for a, b_ in pairs:
        if a not in by or b_ not in by: continue
        la, t = sample_losses(by[a], mask_zero); lb, _ = sample_losses(by[b_], mask_zero)
        ra, rb = np.sqrt(la.mean()), np.sqrt(lb.mean())
        # bootstrap the RMSE difference with the same blocks for both models
        n = len(la); nb = int(np.ceil(n / block))
        starts = RNG.integers(0, n - block + 1, size=(4000, nb))
        idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(4000, -1)[:, :n]
        diff = np.sqrt(la[idx].mean(1)) - np.sqrt(lb[idx].mean(1))
        lo, hi = np.percentile(diff, [2.5, 97.5])
        dm, pv = dm_test(la - lb, lag=block)
        rows.append({'A': a, 'B': b_, 'RMSE_A': ra, 'RMSE_B': rb, 'ΔRMSE (A−B)': ra - rb,
                     'CI95_low': lo, 'CI95_high': hi, 'P(A better)': float((diff < 0).mean()),
                     'DM_stat': dm, 'DM_p': pv})
    return pd.DataFrame(rows)


# ---------- helpers that run a model ----------
@torch.no_grad()
def predict_split(model, b, cfg, split, perturb=None, seed=0):
    ds = TrafficWindows(b, split, cfg['seq_len'], cfg['horizon'])
    L = cfg['seq_len']; rng = np.random.default_rng(seed)
    X = np.stack([b.values[s - L:s] for s in ds.starts]).astype(np.float32)   # raw
    if perturb is not None:
        X = perturb(X, rng)
    Xs = (X - b.means) / b.stds
    TF = np.stack([b.time_features[s - L:s] for s in ds.starts])
    Y = np.stack([b.values[s:s + cfg['horizon']] for s in ds.starts])
    out = []
    for i in range(0, len(Xs), 256):
        x = torch.from_numpy(Xs[i:i + 256])[..., None]
        tf = torch.from_numpy(TF[i:i + 256])
        out.append(model(x, tf).numpy())
    P = np.concatenate(out) * b.stds + b.means
    return Y, P, ds.starts, X


def robustness(by, ds, mask_zero):
    names = [n for n in ('gru', 'tgcn', 'tgcn_selfsep', 'gru_res_time', 'tgcn_res_time') if n in by]
    gauss = [0, 1, 2, 4, 8]; fail = [0, 5, 10, 20, 30]
    rows = []
    for name in names + ['naive']:
        if name != 'naive':
            r = [x for x in by[name] if x['seed'] == 42] or by[name]
            model, b, cfg = load_model(r[0]['dir'])
        for kind, levels in (('gauss', gauss), ('failure', fail)):
            for lv in levels:
                if kind == 'gauss':
                    pert = (lambda X, rng, s=lv: X + rng.normal(0, s, X.shape).astype(np.float32)) if lv else None
                else:
                    pert = (lambda X, rng, q=lv: np.where(rng.random(X.shape) < q / 100, 0, X).astype(np.float32)) if lv else None
                if name == 'naive':
                    if 'b0' not in locals():
                        b0 = bundle_for(ds)
                    Lh = 12; H = 12 if ds == 'losloop' else 4
                    st = TrafficWindows(b0, 'test', Lh, H).starts
                    X = np.stack([b0.values[s - Lh:s] for s in st]).astype(np.float32)
                    if pert is not None: X = pert(X, np.random.default_rng(0))
                    Y = np.stack([b0.values[s:s + H] for s in st]); P = np.repeat(X[:, -1:], H, 1)
                else:
                    Y, P, _, _ = predict_split(model, b, cfg, 'test', pert)
                m = metrics(Y, P, mask_zero=mask_zero)
                rows.append({'model': name, 'perturbation': kind, 'level': lv, 'RMSE': m['RMSE'], 'MAE': m['MAE']})
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, kind, xl in ((axes[0], 'gauss', 'Độ lệch chuẩn nhiễu Gauss thêm vào đầu vào'),
                         (axes[1], 'failure', '% giá trị đầu vào bị mất (đặt = 0)')):
        for i, name in enumerate(['naive'] + names):
            s = df[(df.model == name) & (df.perturbation == kind)]
            ax.plot(s.level, s.RMSE, marker='o', ms=4, color=ps.color(name, i), label=ps.label(name),
                    ls='--' if name == 'naive' else '-')
        ax.set(xlabel=xl, ylabel='RMSE test (mọi tầm dự báo)')
    axes[0].set_title('Nhiễu đo đạc'); axes[1].set_title('Cảm biến hỏng ngẫu nhiên')
    axes[0].legend(loc='upper left', fontsize=8)
    fig.suptitle(f'{ds}: độ bền của mô hình khi dữ liệu đầu vào bị nhiễu', fontsize=11.5)
    fig.tight_layout(); fig.savefig(FIG / f'rel_robustness_{ds}.png'); plt.close(fig)
    return df


def conformal(by, ds, mask_zero, alpha=0.1):
    rows = []
    for name in ('gru', 'tgcn', 'gru_res_time', 'tgcn_res_time'):
        if name not in by: continue
        r = [x for x in by[name] if x['seed'] == 42] or by[name]
        model, b, cfg = load_model(r[0]['dir'])
        Yv, Pv, _, _ = predict_split(model, b, cfg, 'val')
        Yt, Pt, st, _ = predict_split(model, b, cfg, 'test')
        rv = np.abs(Yv - Pv); rt = np.abs(Yt - Pt)
        n = len(rv); lvl = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
        ts = b.timestamps
        for mode in ('per_horizon', 'per_horizon_node'):
            if mode == 'per_horizon':
                q = np.quantile(rv.reshape(n, rv.shape[1], -1).transpose(1, 0, 2).reshape(rv.shape[1], -1),
                                lvl, axis=1)[None, :, None]
            else:
                q = np.quantile(rv, lvl, axis=0)[None]
            cover = rt <= q
            valid = (Yt != 0) if mask_zero else np.ones_like(Yt, bool)
            for k in sorted({2, 5, 11} & set(range(Yt.shape[1]))) or range(Yt.shape[1]):
                tt = ts[st + k]
                h = tt.hour + tt.minute / 60
                peak = np.asarray((tt.dayofweek < 5) & (((h >= 7) & (h < 9)) | ((h >= 16) & (h < 19))))
                c = cover[:, k]; v = valid[:, k]
                rows.append({'model': name, 'mode': mode, 'horizon_min': (k + 1) * b.interval_min,
                             'coverage_all_%': 100 * c[v].mean(),
                             'coverage_peak_%': 100 * c[peak[:, None] & v].mean() if peak.any() else np.nan,
                             'coverage_offpeak_%': 100 * c[~peak[:, None] & v].mean(),
                             'mean_width': float((2 * np.broadcast_to(q[:, k], c.shape))[v].mean())})
    return pd.DataFrame(rows)


def interpolation_check(by, ds):
    """Los-loop: targets on >=6-point straight segments are likely interpolated."""
    b = bundle_for(ds)
    v = b.values; T, N = v.shape
    d2 = np.abs(np.diff(v, 2, axis=0)) < 1e-3
    lin = np.zeros_like(v, bool)
    for j in range(N):
        run = 0
        for t in range(T - 2):
            if d2[t, j]: run += 1
            else:
                if run >= 4: lin[t - run + 1:t + 2, j] = True
                run = 0
    rows = []
    for name in ('naive', 'linear', 'gru', 'tgcn', 'tgcn_res_time'):
        if name not in by: continue
        res_all, res_obs = [], []
        for r in by[name]:
            yt, yp, t = load_pred(r['dir'])
            idx = t[:, None] + np.arange(yt.shape[1])[None]
            m = ~lin[idx]                                           # observed targets
            e = yp - yt
            res_all.append(np.sqrt((e ** 2).mean())); res_obs.append(np.sqrt((e[m] ** 2).mean()))
        rows.append({'model': name, 'RMSE_all_targets': np.mean(res_all),
                     'RMSE_observed_only': np.mean(res_obs),
                     'share_interpolated_test_%': 100 * (1 - m.mean())})
    return pd.DataFrame(rows), float(lin.mean())


def bias_by_level(by, ds):
    bins = [0, 20, 30, 40, 50, 60, 80] if ds == 'losloop' else [0.1, 5, 10, 15, 20, 30, 90]
    rows = []
    for name in ('naive', 'gru', 'tgcn', 'tgcn_res_time'):
        if name not in by: continue
        yt, yp, t = load_pred([x for x in by[name] if x['seed'] in (0, 42)][0]['dir'])
        k = min(2, yt.shape[1] - 1)
        y = yt[:, k].ravel(); e = (yp[:, k] - yt[:, k]).ravel()
        cat = pd.cut(y, bins)
        g = pd.DataFrame({'bin': cat, 'e': e}).groupby('bin', observed=True)['e']
        for bb, s in g:
            rows.append({'model': name, 'speed_bin': str(bb), 'n': len(s), 'mean_error(bias)': s.mean(),
                         'MAE': s.abs().mean()})
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    names = list(dict.fromkeys(df.model))
    labels = list(dict.fromkeys(df.speed_bin)); x = np.arange(len(labels)); w = .8 / len(names)
    for i, name in enumerate(names):
        s = df[df.model == name].set_index('speed_bin').reindex(labels)
        ax.bar(x + (i - (len(names) - 1) / 2) * w, s['mean_error(bias)'], w * .92,
               color=ps.color(name, i), label=ps.label(name))
    ax.axhline(0, color=ps.INK2, lw=.8)
    ax.set(xticks=x, xticklabels=labels, xlabel='Tốc độ thực tế (khoảng)', ylabel='Sai số TB (dự báo − thực)',
           title=f'{ds}: độ lệch theo mức tốc độ (dự báo 15 phút) — dương = dự báo quá lạc quan')
    ax.legend(fontsize=8, loc='upper right'); fig.tight_layout()
    fig.savefig(FIG / f'rel_bias_by_level_{ds}.png'); plt.close(fig)
    return df


def main():
    p = argparse.ArgumentParser(); p.add_argument('--dataset', default='losloop')
    p.add_argument('--skip-model-runs', action='store_true')
    ds = p.parse_args().dataset
    a = p.parse_args()
    mask_zero = ds == 'sztaxi'
    by = runs_by_name(ds)
    st = seed_table(by); st.to_csv(TAB / f'rel_seeds_{ds}.csv', index=False)
    print(st.round(4).to_string())
    block = 12 if ds == 'losloop' else 4
    pairs = [('tgcn', 'gru'), ('tgcn', 'naive'), ('gru', 'naive'), ('tgcn', 'linear'),
             ('tgcn_res_time', 'tgcn'), ('tgcn_res_time', 'naive'), ('tgcn_res_time', 'gru_res_time'),
             ('tgcn_res_time', 'linear'), ('gru_res_time', 'gru'), ('gru_res_time', 'naive'),
             ('tgcn_selfsep', 'gru'), ('tgcn_selfsep', 'tgcn'),
             ('tgcn_selfsep_res_time', 'tgcn_res_time'), ('tgcn_selfsep_res_time', 'gru_res_time'),
             ('ensemble_tgcn_res_time', 'tgcn_res_time'), ('ensemble_gru_res_time', 'gru_res_time'),
             ('ensemble_mix', 'ensemble_gru_res_time')]
    pw = pairwise(by, pairs, block, mask_zero); pw.to_csv(TAB / f'rel_pairwise_{ds}.csv', index=False)
    print(pw.round(4).to_string())
    if ds == 'losloop':
        ic, share = interpolation_check(by, ds); ic.to_csv(TAB / f'rel_interpolation_{ds}.csv', index=False)
        print('interpolated share (all data)', round(share, 4)); print(ic.round(4).to_string())
    bl = bias_by_level(by, ds); bl.to_csv(TAB / f'rel_bias_by_level_{ds}.csv', index=False)
    print(bl.round(3).to_string())
    if not a.skip_model_runs:
        cf = conformal(by, ds, mask_zero); cf.to_csv(TAB / f'rel_conformal_{ds}.csv', index=False)
        print(cf.round(3).to_string())
        rb = robustness(by, ds, mask_zero); rb.to_csv(TAB / f'rel_robustness_{ds}.csv', index=False)
        print(rb.pivot_table(index=['perturbation', 'level'], columns='model', values='RMSE').round(3).to_string())


if __name__ == '__main__':
    main()
