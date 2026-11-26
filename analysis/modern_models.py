"""Section 13 of the report — newer architectures, LightGBM and profile features.

Compares every model on the same test windows, builds ensembles, runs paired
significance tests against the previous best (GRU-RT ensemble), splits errors by
traffic state and summarises the LightGBM feature importance.

    python analysis/modern_models.py --dataset losloop
"""
import argparse
import json
from collections import defaultdict
import numpy as np
import pandas as pd
from common import ROOT, FIG, TAB, run_dirs, load_pred, bundle_for, val_rmse, df_to_markdown
from tgcn.metrics import horizon_metrics
from tgcn import plotstyle as ps
from reliability import sample_losses, dm_test, RNG
import matplotlib.pyplot as plt

LABEL = {'gru': 'GRU', 'tgcn': 'T-GCN', 'gru_res_time': 'GRU-RT', 'tgcn_res_time': 'T-GCN-RT',
         'tgcn_selfsep': 'T-GCN tách self', 'tgcn_selfsep_res_time': 'T-GCN-RTS',
         'stid': 'STID', 'stid_res': 'STID + residual', 'dcrnn': 'DCRNN-lite', 'dcrnn_res': 'DCRNN-lite + residual',
         'gwnet': 'Graph WaveNet', 'gwnet_res': 'Graph WaveNet + residual',
         'gru_rtp': 'GRU-RTP (+ hồ sơ giờ)', 'tgcn_rtsp': 'T-GCN-RTSP (+ hồ sơ giờ)', 'lgbm': 'LightGBM (đặc trưng KG-TG)',
         'lgbm_v2': 'LightGBM v2 (+ bất thường láng giềng/toàn mạng)',
         'lgbm_best': 'LightGBM v2, 127 lá (chọn theo val)',
         'staeformer': 'STAEformer', 'staeformer_full': 'STAEformer (full)', 'staeformer_full_res': 'STAEformer full + res',
         'gwnet_full': 'Graph WaveNet (full)', 'gwnet_full_res': 'Graph WaveNet full + res',
         'naive': 'Persistence', 'ha_window': 'HA (TB cửa sổ)', 'linear': 'Hồi quy tuyến tính'}
FAMILY = {'naive': 'baseline', 'ha_window': 'baseline', 'linear': 'baseline', 'gru': 'paper', 'tgcn': 'paper',
          'gru_res_time': 'phần 1', 'tgcn_res_time': 'phần 1', 'tgcn_selfsep': 'phần 1', 'tgcn_selfsep_res_time': 'phần 1',
          'lgbm': 'khai phá dữ liệu', 'lgbm_v2': 'khai phá dữ liệu', 'lgbm_best': 'khai phá dữ liệu', 'gru_rtp': 'hồ sơ giờ', 'tgcn_rtsp': 'hồ sơ giờ'}
GROUPS = ('baselines', 'main', 'improve', 'variants', 'modern', 'modern_gpu')
KEEP = ['naive', 'linear', 'gru', 'tgcn', 'gru_res_time', 'tgcn_res_time', 'tgcn_selfsep', 'tgcn_selfsep_res_time',
        'stid', 'stid_res', 'dcrnn', 'dcrnn_res', 'gwnet', 'gwnet_res', 'gwnet_full', 'gwnet_full_res',
        'staeformer', 'staeformer_full', 'staeformer_full_res', 'gru_rtp', 'tgcn_rtsp', 'lgbm', 'lgbm_v2', 'lgbm_best']


def lab(n):
    if n.startswith('gpu_'):
        return GPU_LABEL.get(n[4:], n[4:]) + ' (GPU)'
    if n.startswith('ens_'):
        n = n[4:]
    if n.startswith('lgbm_plus_'):
        return f'Ensemble LightGBM + {LABEL.get(n[10:], n[10:])}'
    if n in ('lgbm', 'gru_res_time', 'gru_rtp') and n in ENS_ONLY:
        return 'Ensemble ' + LABEL[n] + ' (3 seed)'
    return LABEL.get(n, ps.label(n))


ENS_ONLY = set()
GPU_LABEL = {'gwnet': 'Graph WaveNet', 'gwnet_res': 'Graph WaveNet + residual', 'gwnet_full': 'Graph WaveNet cỡ đầy đủ',
             'gwnet_full_res': 'Graph WaveNet cỡ đầy đủ + residual', 'staeformer': 'STAEformer',
             'staeformer_full': 'STAEformer cỡ đầy đủ', 'staeformer_full_res': 'STAEformer cỡ đầy đủ + residual'}


def collect(ds):
    """Runs by model name; runs made on the GPU (group modern_gpu, if their folders were copied
    into results/) are kept apart under 'gpu_<name>'."""
    by = defaultdict(list)
    for r in run_dirs(ds):
        if r['group'] in GROUPS and r['name'] in KEEP:
            by[('gpu_' if r['group'] == 'modern_gpu' else '') + r['name']].append(r)
    return by


def mean_pred(rs):
    ps_ = [load_pred(r['dir']) for r in rs]
    return ps_[0][0], np.mean([p[1] for p in ps_], axis=0), ps_[0][2]


def save_ensemble(ds, name, yt, yp, t, members, interval):
    sc = horizon_metrics(yt, yp, interval)
    if (yt == 0).any():
        sc['masked_nonzero'] = horizon_metrics(yt, yp, interval, mask_zero=True)
    out = ROOT / 'results' / ds / 'ensemble' / name
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / 'test_predictions.npz', y_true=yt, y_pred=yp.astype(np.float32), time_indices=t)
    from tgcn.utility import save_json
    save_json({**sc, 'dataset': ds, 'model': f'ensemble_{name}', 'seed': 0, 'caution': '', 'members': members},
              out / 'metrics.json')
    return sc


def pair_test(la, lb, block):
    n = len(la); nb = int(np.ceil(n / block))
    starts = RNG.integers(0, n - block + 1, size=(4000, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(4000, -1)[:, :n]
    diff = np.sqrt(la[idx].mean(1)) - np.sqrt(lb[idx].mean(1))
    dm, pv = dm_test(la - lb, lag=block)
    return np.sqrt(la.mean()) - np.sqrt(lb.mean()), *np.percentile(diff, [2.5, 97.5]), pv


def per_sample_loss(yt, yp, mask_zero):
    e2 = (yp - yt) ** 2
    if mask_zero:
        m = yt != 0
        return (e2 * m).sum((1, 2)) / m.sum((1, 2)).clip(1)
    return e2.mean((1, 2))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--dataset', default='losloop')
    ds = ap.parse_args().dataset
    mask_zero = ds == 'sztaxi'
    b = bundle_for(ds)
    iv = b.interval_min
    by = collect(ds)
    if 'tgcn' not in by or 'gru' not in by:
        raise SystemExit(f'results/{ds}/main/(gru|tgcn) not found: this script needs the full set of runs '
                         '(results/ is not in git). Tables left unchanged.')
    view = (lambda m: m.get('masked_nonzero', m)) if mask_zero else (lambda m: m)
    # ---- main table
    rows = []
    for name in KEEP + [f'gpu_{c}' for c in GPU_LABEL]:
        rs = by.get(name)
        if not rs: continue
        ms = [view(r['metrics']) for r in rs]
        vr = [val_rmse(r['dir'], r['metrics'].get('best_epoch')) if not r['name'].startswith('lgbm')
              else json.loads((r['dir'] / 'training_summary.json').read_text())['val_rmse_raw'] for r in rs]
        f = lambda key, k=None: np.array([(m['overall'] if k is None else m['per_horizon'][k])[key] for m in ms])
        ks = list(ms[0]['per_horizon'])
        k15, k60 = (ks[2], ks[11]) if len(ks) >= 12 else (ks[0], ks[-1])
        summ = rs[0]['summary']
        rows.append({'name': name, 'Mô hình': lab(name), 'seed': len(rs),
                     'nhóm': 'hiện đại (GPU)' if name.startswith('gpu_') else FAMILY.get(name, 'hiện đại'),
                     'tham số': summ.get('n_parameters'),
                     'RMSE val': np.nanmean(vr), 'RMSE': f('RMSE').mean(), 'RMSE_sd': f('RMSE').std(ddof=1) if len(rs) > 1 else np.nan,
                     'MAE': f('MAE').mean(), 'MAE_sd': f('MAE').std(ddof=1) if len(rs) > 1 else np.nan,
                     'RMSE 15': f('RMSE', k15).mean(), 'MAE 15': f('MAE', k15).mean(),
                     'RMSE 60': f('RMSE', k60).mean(), 'MAE 60': f('MAE', k60).mean(),
                     'phút train/run': np.nanmean([r['summary'].get('train_seconds', np.nan) for r in rs]) / 60})
    df = pd.DataFrame(rows)
    # ---- ensembles
    ens = {}
    LG = 'lgbm_best' if len(by.get('lgbm_best', [])) >= 3 else 'lgbm'
    if LG in by and len(by[LG]) > 1:
        ens[LG] = mean_pred(by[LG])
    pref_deep = 'gru_rtp' if 'gru_rtp' in by else 'gru_res_time'
    if LG in by and pref_deep in by:
        yt, a, t = mean_pred(by[LG]); _, bb, _ = mean_pred(by[pref_deep])
        ens['lgbm_plus_' + pref_deep] = (yt, (a + bb) / 2, t)
    blend = [r for r in run_dirs(ds) if r['group'] == 'ensemble' and r['name'] == 'blend']
    if 'gru_res_time' in by:
        ens['gru_res_time'] = mean_pred(by['gru_res_time'])
    if 'gru_rtp' in by:
        ens['gru_rtp'] = mean_pred(by['gru_rtp'])
    for name, (yt, yp, t) in ens.items():
        sc = save_ensemble(ds, name, yt, yp, t, [], iv)
        if name in by and len(by[name]) < 2:      # a one-run "ensemble" is that run: no extra row
            continue
        m = view(sc); ks = list(m['per_horizon'])
        k15, k60 = (ks[2], ks[11]) if len(ks) >= 12 else (ks[0], ks[-1])
        label = (f'Ensemble LightGBM + {LABEL.get(name[10:], name[10:])}' if name.startswith('lgbm_plus_')
                 else f'Ensemble {LABEL.get(name, name)} (3 seed)')
        rows.append({'name': f'ens_{name}', 'Mô hình': label,
                     'nhóm': 'ensemble', 'seed': 3, 'tham số': None, 'RMSE val': np.nan,
                     'RMSE': m['overall']['RMSE'], 'RMSE_sd': np.nan, 'MAE': m['overall']['MAE'], 'MAE_sd': np.nan,
                     'RMSE 15': m['per_horizon'][k15]['RMSE'], 'MAE 15': m['per_horizon'][k15]['MAE'],
                     'RMSE 60': m['per_horizon'][k60]['RMSE'], 'MAE 60': m['per_horizon'][k60]['MAE'],
                     'phút train/run': np.nan})
    if blend:
        m = view(blend[0]['metrics']); ks = list(m['per_horizon'])
        k15, k60 = (ks[2], ks[11]) if len(ks) >= 12 else (ks[0], ks[-1])
        short = {'lgbm_best': 'LightGBM', 'gwnet_res': 'Graph WaveNet-res', 'gru_res_time': 'GRU-RT',
                 'tgcn_rtsp': 'T-GCN-RTSP', 'gru_rtp': 'GRU-RTP', 'stid_res': 'STID-res', 'gru': 'GRU'}
        rows.append({'name': 'ens_blend', 'Mô hình': 'Trộn theo tầm: ' + ' + '.join(short.get(x, x) for x in blend[0]['metrics'].get('members', [])),
                     'nhóm': 'ensemble', 'seed': 3, 'tham số': None, 'RMSE val': np.nan,
                     'RMSE': m['overall']['RMSE'], 'RMSE_sd': np.nan, 'MAE': m['overall']['MAE'], 'MAE_sd': np.nan,
                     'RMSE 15': m['per_horizon'][k15]['RMSE'], 'MAE 15': m['per_horizon'][k15]['MAE'],
                     'RMSE 60': m['per_horizon'][k60]['RMSE'], 'MAE 60': m['per_horizon'][k60]['MAE'], 'phút train/run': np.nan})
    # ---- configurations run on the user's GPU (per-run metrics only; see analysis/gpu_results.py)
    gs = TAB / f'modern_gpu_summary_{ds}.csv'
    if gs.is_file():
        for _, g in pd.read_csv(gs).iterrows():
            if f'gpu_{g.config}' in by:          # prediction files available -> already in the table
                continue
            nz = mask_zero and 'RMSE ≠0' in g
            rows.append({'name': f'gpu_{g.config}', 'Mô hình': GPU_LABEL.get(g.config, g.config) + ' (GPU)',
                         'nhóm': 'hiện đại (GPU)', 'seed': int(g.seed), 'tham số': g['tham số'], 'RMSE val': np.nan,
                         'RMSE': g['RMSE ≠0'] if nz else g.RMSE, 'RMSE_sd': g['RMSE ≠0_sd'] if nz else g.RMSE_sd,
                         'MAE': g['MAE ≠0'] if nz else g.MAE, 'MAE_sd': g['MAE ≠0_sd'] if nz else g.MAE_sd,
                         # per-step errors on targets != 0 were not exported by the GPU runs
                         'RMSE 15': np.nan if mask_zero else g['RMSE 15'], 'MAE 15': np.nan if mask_zero else g['MAE 15'],
                         'RMSE 60': np.nan if mask_zero else g['RMSE 60'], 'MAE 60': np.nan if mask_zero else g['MAE 60'],
                         'phút train/run': np.nan})
    df = pd.DataFrame(rows).sort_values('RMSE')
    df.to_csv(TAB / f'modern_results_{ds}.csv', index=False)
    # LightGBM configurations (seed 42), chosen by validation
    tune = []
    for tag, desc in (('lgbm', 'v1, 63 lá, lr 0.05'), ('lgbm_v2', 'v2, 63 lá, lr 0.05'),
                      ('lgbm_v2_num_leaves31', 'v2, 31 lá'), ('lgbm_v2_num_leaves127', 'v2, 127 lá'),
                      ('lgbm_v2_learning_rate0.03', 'v2, lr 0.03')):
        d = ROOT / 'results' / ds / 'modern' / tag / 'seed_42'
        if (d / 'metrics.json').is_file():
            mm = view(json.loads((d / 'metrics.json').read_text()))
            vv = json.loads((d / 'training_summary.json').read_text())['val_rmse_raw']
            tune.append({'Cấu hình LightGBM (seed 42)': desc, 'RMSE val': round(vv, 3),
                         'RMSE test': round(mm['overall']['RMSE'], 3), 'MAE test': round(mm['overall']['MAE'], 3)})
    if tune:
        pd.DataFrame(tune).to_csv(TAB / f'lgbm_tuning_{ds}.csv', index=False)
    print(df.drop(columns=['name']).round(3).to_string(index=False))
    # ---- paired tests against the previous best and against LightGBM
    block = 12 if ds == 'losloop' else 4
    ref_name = 'ensemble GRU-RT' if len(by.get('gru_res_time', [])) > 1 else 'GRU-RT (1 seed)'
    ref = ens.get('gru_res_time')
    tests = []
    cand = {**{k: mean_pred(v) for k, v in by.items() if k in ('lgbm', 'lgbm_best', 'gru_rtp', 'tgcn_rtsp', 'stid_res', 'stid',
                                                                 'dcrnn', 'dcrnn_res', 'gwnet', 'gwnet_res') or k.startswith('gpu_')},
            **{f'ens_{k}': v for k, v in ens.items() if k != 'gru_res_time'},
            **({'ens_blend': load_pred(blend[0]['dir'])} if blend else {})}
    if ref is not None:
        lb = per_sample_loss(ref[0], ref[1], mask_zero)
        for k, (yt, yp, _) in cand.items():
            d, lo, hi, pv = pair_test(per_sample_loss(yt, yp, mask_zero), lb, block)
            a_lab = (lab(k) + (' (TB 3 seed)' if len(by.get(k, [])) > 1 else '')) if not k.startswith('ens_') else \
                ('Trộn theo tầm (fit trên val)' if k == 'ens_blend' else
                 f'Ensemble LightGBM + {LABEL.get(k[14:], k[14:])}' if k.startswith('ens_lgbm_plus_')
                 else f'Ensemble {LABEL.get(k[4:], k[4:])} (3 seed)')
            tests.append({'A': a_lab,
                          'B': ref_name, 'ΔRMSE (A−B)': d, 'CI95_low': lo, 'CI95_high': hi, 'DM_p': pv})
    if 'gru_rtp' in by and 'gru_res_time' in by:
        a = per_sample_loss(*mean_pred(by['gru_rtp'])[:2], mask_zero)
        bq = per_sample_loss(*mean_pred(by['gru_res_time'])[:2], mask_zero)
        d, lo, hi, pv = pair_test(a, bq, block)
        tests.append({'A': f"GRU-RTP ({len(by['gru_rtp'])} seed)", 'B': f"GRU-RT ({len(by['gru_res_time'])} seed)", 'ΔRMSE (A−B)': d, 'CI95_low': lo, 'CI95_high': hi, 'DM_p': pv})
    tdf = pd.DataFrame(tests)
    tdf.to_csv(TAB / f'modern_tests_{ds}.csv', index=False)
    print(tdf.round(4).to_string(index=False))
    # ---- error by traffic state (15 and 60 min)
    sub_rows = []
    show = {k: v for k, v in [('naive', None), ('tgcn', None), ('gru_res_time', None), ('gru_rtp', None), (LG, None)]
            if k in by or k == 'naive'}
    preds = {}
    for k in show:
        if k == 'naive':
            r = [x for x in run_dirs(ds) if x['group'] == 'baselines' and x['name'] == 'naive']
            preds[k] = load_pred(r[0]['dir'])
        else:
            preds[k] = mean_pred(by[k])
    if blend:
        preds['blend'] = load_pred(blend[0]['dir'])
    for k, (yt, yp, t) in preds.items():
        for hk, step in (('15 phút', 15 // iv - 1), ('60 phút', 60 // iv - 1)):
            ts = b.timestamps[t + step]
            h = ts.hour + ts.minute / 60
            peak = np.asarray((ts.dayofweek < 5) & (((h >= 7) & (h < 9)) | ((h >= 16) & (h < 19))))
            y = yt[:, step]; e = np.abs(yp[:, step] - y)
            valid = (y != 0) if mask_zero else np.ones_like(y, bool)
            last = b.values[t - 1]
            thr = 40 if ds == 'losloop' else 10
            for sname, sel in (('tất cả', valid), ('cao điểm', peak[:, None] & valid), ('ngoài cao điểm', ~peak[:, None] & valid),
                               (f'kẹt xe (v<{thr})', (y < thr) & valid), ('biến động mạnh (Δ>10)', (np.abs(y - last) > 10) & valid)):
                sub_rows.append({'model': k, 'horizon': hk, 'subset': sname, 'MAE': float(e[sel].mean())})
    sdf = pd.DataFrame(sub_rows)
    sdf.to_csv(TAB / f'modern_subsets_{ds}.csv', index=False)
    print(sdf.pivot_table(index=['horizon', 'model'], columns='subset', values='MAE', sort=False).round(3).to_string())
    # ---- figures
    fig, ax = plt.subplots(figsize=(10, 0.36 * len(df) + 1.3))
    d2 = df.iloc[::-1]
    y = np.arange(len(d2))
    modern = ('stid', 'stid_res', 'dcrnn', 'dcrnn_res', 'gwnet', 'gwnet_res', 'staeformer')
    colors = [ps.SERIES[6] if n.startswith('ens') else ps.SERIES[1] if n.startswith('lgbm') else
              ps.SERIES[2] if n in ('gru_rtp', 'tgcn_rtsp') else ps.SERIES[4] if n in modern or n.startswith('gpu_') else
              ps.NEUTRAL if n in ('naive', 'linear') else ps.SERIES[0] for n in d2['name']]
    bars = ax.barh(y, d2['RMSE'], color=colors, height=.7)
    for bar, n in zip(bars, d2['name']):
        if n.startswith('gpu_'):
            bar.set_hatch('///'); bar.set_edgecolor(ps.SURFACE)
    for yi, v in zip(y, d2['RMSE']):
        ax.text(v + 0.03, yi, f'{v:.2f}', va='center', fontsize=7.5, color=ps.INK2)
    ax.set(yticks=y, yticklabels=['T-GCN gốc (baseline)' if n == 'tgcn' else m for n, m in zip(d2['name'], d2['Mô hình'])],
           xlabel='RMSE test (TB mọi tầm)' + (' — mục tiêu ≠ 0' if mask_zero else ''),
           title=f'{ds}: mọi mô hình trên cùng tập test')
    for tick, n in zip(ax.get_yticklabels(), d2['name']):
        if n == 'tgcn':
            tick.set_fontweight('bold'); tick.set_color(ps.INK)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c, label=l) for c, l in ((ps.SERIES[6], 'ensemble / trộn'), (ps.SERIES[1], 'LightGBM'),
                                                            (ps.SERIES[4], 'kiến trúc hiện đại (gạch chéo: chạy trên GPU)'),
                                                            (ps.SERIES[2], 'mạng + hồ sơ giờ'),
                                                            (ps.SERIES[0], 'GRU / T-GCN và cải tiến phần 1'),
                                                            (ps.NEUTRAL, 'baseline'))],
              loc='upper right', fontsize=8, frameon=True, facecolor=ps.SURFACE, edgecolor=ps.GRID)
    lo = df['RMSE'].min() * 0.9
    ax.set_xlim(lo, df['RMSE'].max() * 1.04); ax.grid(axis='y', visible=False)
    fig.tight_layout(); fig.savefig(FIG / f'modern_ranking_{ds}.png'); plt.close(fig)
    # horizon curves
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    curves = [('gru', 'GRU', ps.SERIES[0], '-'), ('tgcn', 'T-GCN', ps.SERIES[1], '-'),
              ('gru_res_time', 'GRU-RT (TB 3 seed)', ps.MODEL_COLORS['gru_res_time'], '-'),
              ('stid_res', 'STID + residual', ps.SERIES[4], '-'), ('dcrnn_res', 'DCRNN-lite + residual', ps.SERIES[7], '-'),
              ('gwnet', 'Graph WaveNet', ps.SERIES[5], '-'), ('gru_rtp', 'GRU-RTP', ps.SERIES[2], '-'),
              (LG, 'LightGBM', ps.SERIES[3], '-')]
    for key, name, c, ls in curves:
        if key not in by: continue
        ms = [view(r['metrics']) for r in by[key]]; ks = list(ms[0]['per_horizon'])
        x = [(i + 1) * iv for i in range(len(ks))]
        for ax, met in zip(axes, ('RMSE', 'MAE')):
            ax.plot(x, np.mean([[m['per_horizon'][k][met] for k in ks] for m in ms], 0), color=c, ls=ls, label=name, lw=1.8)
    naive = [x for x in run_dirs(ds) if x['group'] == 'baselines' and x['name'] == 'naive']
    if naive:
        m = view(naive[0]['metrics']); ks = list(m['per_horizon'])
        for ax, met in zip(axes, ('RMSE', 'MAE')):
            ax.plot([(i + 1) * iv for i in range(len(ks))], [m['per_horizon'][k][met] for k in ks], color=ps.NEUTRAL,
                    ls='--', label='Persistence')
    for ax, met in zip(axes, ('RMSE', 'MAE')):
        ax.set(xlabel='Tầm dự báo (phút)', ylabel=met, title=f'{met} theo tầm dự báo')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=5, fontsize=8, bbox_to_anchor=(0.5, -0.1))
    fig.suptitle(f'{ds}: mô hình hiện đại và LightGBM theo tầm dự báo', fontsize=11.5)
    fig.tight_layout(); fig.savefig(FIG / f'modern_horizon_{ds}.png'); plt.close(fig)
    # LightGBM feature importance
    if LG in by:
        imps = [pd.read_csv(r['dir'] / 'feature_importance_gain.csv', index_col=0) for r in by[LG]]
        share = sum(i / i.sum() for i in imps) / len(imps)                     # [features, horizons]
        groups = {'lịch sử chính nút': [c for c in share.index if c.startswith(('lag', 'diff', 'mean', 'std', 'min1', 'max'))],
                  'láng giềng (đồ thị)': [c for c in share.index if c.startswith(('nb', 'self_minus'))],
                  'trạng thái toàn mạng': [c for c in share.index if c.startswith('net_')],
                  'lịch (giờ, thứ)': ['tod', 'tod_sin', 'tod_cos', 'dow', 'weekend'],
                  'hồ sơ theo giờ của cảm biến': ['prof_now', 'dev_from_prof', 'prof_change_k', 'dev_lag2', 'dev_lag4',
                                                  'dev_lag7', 'dev_trend3'],
                  'đặc tính cảm biến': ['sensor_mean', 'sensor_std', 'degree']}
        g = pd.DataFrame({k: share.loc[[c for c in v if c in share.index]].sum() for k, v in groups.items()})
        g.index = [int(i) * iv for i in g.index]
        g.to_csv(TAB / f'lgbm_importance_groups_{ds}.csv')
        top = share.mean(axis=1).sort_values(ascending=False).head(15)
        top.to_csv(TAB / f'lgbm_importance_top_{ds}.csv')
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].barh(np.arange(len(top))[::-1], 100 * top.values, color=ps.SERIES[3])
        axes[0].set(yticks=np.arange(len(top))[::-1], yticklabels=top.index, xlabel='% tổng gain (TB các tầm, các seed)',
                    title='15 đặc trưng quan trọng nhất'); axes[0].grid(axis='y', visible=False)
        bottom = np.zeros(len(g))
        for i, col in enumerate(g.columns):
            axes[1].bar(g.index, 100 * g[col], bottom=bottom, width=iv * 0.8, color=ps.SERIES[i], label=col)
            bottom += 100 * g[col].values
        axes[1].set(xlabel='Tầm dự báo (phút)', ylabel='% gain', title='Nhóm thông tin theo tầm dự báo', ylim=(0, 100))
        axes[1].legend(fontsize=7.5, loc='upper left', bbox_to_anchor=(1.01, 1.0))
        fig.suptitle(f'{ds}: LightGBM dựa vào thông tin gì?', fontsize=11.5)
        fig.tight_layout(); fig.savefig(FIG / f'lgbm_importance_{ds}.png'); plt.close(fig)
        print(g.round(3).to_string()); print((100 * top).round(1).to_string())


if __name__ == '__main__':
    main()
