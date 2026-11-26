"""Check and summarise the runs made on the user's GPU (reports/external/).

Only per-run metric tables are available for these runs (no prediction files), so:
  * summaries are recomputed from the per-run rows and compared with the user's
    own summary file;
  * configurations also trained here on CPU are compared (hardware replication);
  * the authors' "15/30/45/60 min" metrics (errors pooled over steps 1..k) are
    derived exactly from the per-step errors: with the same test windows, every step
    has the same number of targets, so RMSE_1..k = sqrt(mean_j RMSE_j^2),
    MAE_1..k = mean_j MAE_j, and Accuracy / R2 follow from ||Y|| and SST of the
    common test targets (verified to 1e-6 against runs that have both).

    python analysis/gpu_results.py
"""
import numpy as np
import pandas as pd
from common import ROOT, FIG, TAB, run_dirs, bundle_for
from tgcn import plotstyle as ps
import matplotlib.pyplot as plt

EXT = ROOT / 'reports' / 'external'
LABEL = {'gwnet': 'Graph WaveNet (gọn)', 'gwnet_res': 'Graph WaveNet + res (gọn)',
         'gwnet_full': 'Graph WaveNet (đầy đủ)', 'gwnet_full_res': 'Graph WaveNet + res (đầy đủ)',
         'staeformer': 'STAEformer (gọn)', 'staeformer_full': 'STAEformer (đầy đủ)',
         'staeformer_full_res': 'STAEformer + res (đầy đủ)'}
ORDER = list(LABEL)
STEPS = {'losloop': (5, [15, 30, 45, 60]), 'sztaxi': (15, [15, 30, 45, 60])}


def load(ds):
    return pd.read_csv(EXT / f'modern_gpu_runs_{ds}.csv', encoding='utf-8-sig')


FLAGS = {'gwnet': ('gwnet', 'compact', False), 'gwnet_res': ('gwnet', 'compact', True),
         'gwnet_full': ('gwnet', 'full', False), 'gwnet_full_res': ('gwnet', 'full', True),
         'staeformer': ('staeformer', 'compact', False), 'staeformer_full': ('staeformer', 'full', False),
         'staeformer_full_res': ('staeformer', 'full', True)}


def n_params(ds, cfg):
    """Parameter count of a GPU configuration (same constructor as train.py)."""
    from tgcn.models_modern import make_modern
    name, size, res = FLAGS[cfg]
    b = bundle_for(ds)
    m = make_modern(name, b, 12, {'losloop': 12, 'sztaxi': 4}[ds], residual=res, size=size)
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def summary(df, nonzero, ds):
    rows = []
    for cfg in ORDER:
        g = df[df.config == cfg]
        if g.empty:
            continue
        r = {'config': cfg, 'Mô hình (GPU)': LABEL[cfg], 'seed': len(g), 'tham số': n_params(ds, cfg),
             'epoch chọn': f"{int(g.best_epoch.min())}–{int(g.best_epoch.max())}",
             'RMSE': g.RMSE_all.mean(), 'RMSE_sd': g.RMSE_all.std(ddof=1),
             'MAE': g.MAE_all.mean(), 'MAE_sd': g.MAE_all.std(ddof=1),
             'RMSE 15': g.RMSE_15min.mean(), 'MAE 15': g.MAE_15min.mean(),
             'RMSE 60': g.RMSE_60min.mean(), 'MAE 60': g.MAE_60min.mean()}
        if nonzero:
            r.update({'RMSE ≠0': g.RMSE_all_nonzero.mean(), 'RMSE ≠0_sd': g.RMSE_all_nonzero.std(ddof=1),
                      'MAE ≠0': g.MAE_all_nonzero.mean(), 'MAE ≠0_sd': g.MAE_all_nonzero.std(ddof=1)})
        rows.append(r)
    return pd.DataFrame(rows)


def derive_cumulative(ds, df):
    """Paper-style metrics (pooled over steps 1..k) for every external run."""
    yt = np.load(ROOT / 'results' / ds / 'baselines' / 'naive' / 'test_predictions.npz')['y_true'].astype(np.float64)
    iv, mins = STEPS[ds]
    keys = [c[5:] for c in df.columns if c.startswith('RMSE_') and c.endswith('min')]
    keys = sorted(keys, key=lambda k: int(k[:-3]))
    out = []
    for _, r in df.iterrows():
        row = {'config': r.config, 'seed': r.seed}
        for m in mins:
            k = m // iv
            use = [kk for kk in keys if int(kk[:-3]) <= m]
            if len(use) != k:              # some per-step columns missing -> cannot derive
                continue
            rm = np.sqrt(np.mean([r[f'RMSE_{kk}'] ** 2 for kk in use])); ma = np.mean([r[f'MAE_{kk}'] for kk in use])
            Y = yt[:, :k, :]; sse = Y.size * rm ** 2
            row.update({f'RMSE_cum{m}': rm, f'MAE_cum{m}': ma,
                        f'Accuracy_cum{m}': 1 - np.sqrt(sse) / np.linalg.norm(Y),
                        f'R2_cum{m}': 1 - sse / ((Y - Y.mean()) ** 2).sum()})
        out.append(row)
    return pd.DataFrame(out)


def local_counterparts(ds):
    """Same configurations trained here on CPU (group modern)."""
    by = {}
    for r in run_dirs(ds):
        if r['group'] == 'modern' and r['name'] in ('gwnet', 'gwnet_res', 'staeformer'):
            by.setdefault(r['name'], []).append(r['metrics']['overall'])
    return {k: (np.mean([m['RMSE'] for m in v]), np.std([m['RMSE'] for m in v], ddof=1) if len(v) > 1 else np.nan,
                np.mean([m['MAE'] for m in v]), len(v)) for k, v in by.items()}


def main():
    checks = []
    figs = {}
    for ds in ('losloop', 'sztaxi'):
        if not (EXT / f'modern_gpu_runs_{ds}.csv').is_file():
            continue
        df = load(ds)
        nonzero = 'RMSE_all_nonzero' in df.columns
        sm = summary(df, nonzero, ds)
        sm.to_csv(TAB / f'modern_gpu_summary_{ds}.csv', index=False)
        figs[ds] = sm
        print(f'== {ds}\n', sm.drop(columns=['config']).round(3).to_string(index=False))
        # (1) the user's own summary file, if any
        user = EXT / f'user_summary_{ds}.csv'
        if user.is_file():
            u = pd.read_csv(user, encoding='utf-8-sig').set_index('config')
            for _, r in sm.iterrows():
                if r.config in u.index:
                    d = abs(u.loc[r.config, 'RMSE_all_mean'] - r.RMSE)
                    checks.append({'dataset': ds, 'kiểm tra': f'tóm tắt của bạn = tính lại từ từng run ({r.config})',
                                   'kết quả': 'khớp' if d < 1e-9 else f'lệch {d:.2e}'})
        # (2) per-step errors must grow with the horizon (sanity)
        cols = sorted([c for c in df.columns if c.startswith('RMSE_') and c.endswith('min')], key=lambda c: int(c[5:-3]))
        mono = (np.diff(df[cols].to_numpy(), axis=1) > -0.05).all(axis=1)
        checks.append({'dataset': ds, 'kiểm tra': 'RMSE tăng theo tầm dự báo ở mọi run (dung sai 0.05)',
                       'kết quả': f'{int(mono.sum())}/{len(df)} run đạt'})
        # (3) hardware replication
        for cfg, (rm, rsd, ma, n) in local_counterparts(ds).items():
            g = sm[sm.config == cfg]
            if len(g):
                g = g.iloc[0]
                checks.append({'dataset': ds, 'kiểm tra': f'{LABEL[cfg]}: CPU ({n} seed) vs GPU (3 seed)',
                               'kết quả': f'CPU {rm:.3f}' + (f' ± {rsd:.3f}' if np.isfinite(rsd) else '') +
                                          f'; GPU {g.RMSE:.3f} ± {g.RMSE_sd:.3f}'})
        if (ROOT / 'results' / ds / 'baselines' / 'naive' / 'test_predictions.npz').is_file():
            derive_cumulative(ds, df).to_csv(TAB / f'modern_gpu_cumulative_{ds}.csv', index=False)
        else:
            print(f'{ds}: test targets not found (run baselines.py first) - pooled metrics left unchanged')
    ck = pd.DataFrame(checks)
    print(ck.to_string(index=False))
    if not any('CPU' in c for c in ck['kiểm tra']):
        print('no CPU runs in results/ (not in git): checks table and figure left unchanged')
        return
    ck.to_csv(TAB / 'modern_gpu_checks.csv', index=False)
    # figure: every GPU configuration, mean +- sd over 3 seeds, with references
    fig, axes = plt.subplots(1, len(figs), figsize=(6.2 * len(figs), 4.3), squeeze=False)
    refs = {'losloop': [('T-GCN gốc', 'main', 'tgcn', ps.MODEL_COLORS['tgcn']),
                        ('GRU-RT', 'improve', 'gru_res_time', ps.MODEL_COLORS['gru_res_time']),
                        ('LightGBM', 'modern', 'lgbm_best', ps.SERIES[3])],
            'sztaxi': [('T-GCN gốc', 'main', 'tgcn', ps.MODEL_COLORS['tgcn']),
                       ('STID (CPU)', 'modern', 'stid', ps.SERIES[6]),
                       ('LightGBM', 'modern', 'lgbm_best', ps.SERIES[3])]}
    title = {'losloop': 'Los-loop', 'sztaxi': 'SZ-taxi'}
    for ax, (ds, sm) in zip(axes[0], figs.items()):
        key, sd = ('RMSE ≠0', 'RMSE ≠0_sd') if ds == 'sztaxi' else ('RMSE', 'RMSE_sd')
        y = np.arange(len(sm))[::-1]
        colors = [ps.SERIES[4] if c.startswith('stae') else ps.SERIES[5] for c in sm.config]
        ax.errorbar(sm[key], y, xerr=sm[sd], fmt='none', ecolor=ps.NEUTRAL_LIGHT, capsize=3, zorder=1)
        ax.scatter(sm[key], y, c=colors, s=46, zorder=2)
        for lab, g, n, c in refs[ds]:
            vals = [(r['metrics'].get('masked_nonzero', r['metrics']) if ds == 'sztaxi' else r['metrics'])['overall']['RMSE']
                    for r in run_dirs(ds) if r['group'] == g and r['name'] == n]
            if vals:
                v = float(np.mean(vals))
                ax.axvline(v, color=c, ls='--', lw=1.3, label=f'{lab} (CPU, {v:.2f})' if 'CPU' not in lab else f'{lab[:-6]} (CPU, {v:.2f})')
        ax.set(yticks=y, yticklabels=sm['Mô hình (GPU)'],
               xlabel='RMSE test, TB mọi tầm' + (' (mục tiêu ≠ 0)' if ds == 'sztaxi' else ''), title=title[ds])
        ax.grid(axis='y', visible=False)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.17), ncol=3, fontsize=8, frameon=False)
    fig.suptitle('7 cấu hình chạy trên GPU: trung bình ± độ lệch chuẩn 3 seed; đường đứng = mốc tham chiếu chạy trên CPU',
                 fontsize=10.5)
    fig.tight_layout(); fig.savefig(FIG / 'modern_gpu_configs.png'); plt.close(fig)


if __name__ == '__main__':
    main()
