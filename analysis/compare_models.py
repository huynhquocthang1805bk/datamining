"""Phase 3 — aggregate every finished run, compare with the paper, seed ensembles,
and quantify the optimistic bias of selecting the epoch on the test set.

    python analysis/compare_models.py --dataset losloop
"""
import argparse
from collections import defaultdict
import numpy as np
import pandas as pd
from common import (ROOT, FIG, TAB, PAPER, REPORT_MIN, run_dirs, load_pred, bundle_for,
                    steps_for_minutes, fmt, df_to_markdown, val_rmse)
from tgcn.metrics import horizon_metrics, metrics
from tgcn import plotstyle as ps
import matplotlib.pyplot as plt

ORDER = ['naive', 'ha_window', 'ha_tod', 'linear', 'gcn', 'gru', 'tgcn', 'tgcn_time', 'tgcn_res',
         'gru_res_time', 'tgcn_res_time', 'tgcn_res_time_nograph', 'tgcn_plus', 'tgcn_res_time_huber',
         'tgcn_selfsep', 'tgcn_selfsep_res_time', 'tgcn_selfloop1',
         'gru_res_time_rawloss', 'tgcn_res_time_rawloss',
         'tgcn_masked', 'tgcn_res_time_masked',
         'ensemble_gru', 'ensemble_tgcn', 'ensemble_tgcn_selfsep', 'ensemble_gru_res_time', 'ensemble_tgcn_res_time',
         'ensemble_tgcn_selfsep_res_time', 'ensemble_gru_res_time_rawloss', 'ensemble_mix']
ENSEMBLE_OF = ['gru', 'tgcn', 'tgcn_selfsep', 'gru_res_time', 'tgcn_res_time', 'tgcn_selfsep_res_time',
               'gru_res_time_rawloss']
# heterogeneous ensemble: temporal-only + spatio-temporal family, equal weight per family
MIX = ['gru_res_time', 'tgcn_selfsep_res_time']


def mix_ensemble(by, dataset, interval):
    fams = [by.get(n, []) for n in MIX]
    if any(len(f) < 1 for f in fams):
        return None
    fam_preds = []
    for f in fams:
        ps_ = [load_pred(r['dir']) for r in f]
        fam_preds.append(np.mean([q[1] for q in ps_], axis=0))
    yt, _, t = load_pred(fams[0][0]['dir'])
    yp = np.mean(fam_preds, axis=0)
    sc = horizon_metrics(yt, yp, interval)
    if (yt == 0).any():
        sc['masked_nonzero'] = horizon_metrics(yt, yp, interval, mask_zero=True)
    out = ROOT / 'results' / dataset / 'ensemble' / 'mix'
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / 'test_predictions.npz', y_true=yt, y_pred=yp.astype(np.float32), time_indices=t)
    from tgcn.utility import save_json
    save_json({**sc, 'dataset': dataset, 'model': 'ensemble_mix', 'seed': 0, 'caution': '',
               'members': [str(r['dir']) for f in fams for r in f]}, out / 'metrics.json')
    return sc


def key_of(r):
    return r['name'] if r['group'] in ('baselines', 'main', 'improve', 'variants') else f"{r['name']}"


def collect(dataset, use_masked=False):
    runs = run_dirs(dataset)
    by = defaultdict(list)
    for r in runs:
        if r['group'] in ('tuning', 'ensemble', 'paper_protocol', 'logs'):
            continue
        by[key_of(r)].append(r)
    return runs, by


def ensemble(by, name, dataset, interval):
    rs = by.get(name, [])
    if len(rs) < 2:
        return None
    preds = [load_pred(r['dir']) for r in rs]
    yt = preds[0][0]; yp = np.mean([p[1] for p in preds], axis=0)
    sc = horizon_metrics(yt, yp, interval)
    if (yt == 0).any():
        sc['masked_nonzero'] = horizon_metrics(yt, yp, interval, mask_zero=True)
    out = ROOT / 'results' / dataset / 'ensemble' / name
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / 'test_predictions.npz', y_true=yt, y_pred=yp.astype(np.float32),
                        time_indices=preds[0][2])
    from tgcn.utility import save_json
    save_json({**sc, 'dataset': dataset, 'model': f'ensemble_{name}', 'seed': 0, 'caution': '',
               'members': [str(r['dir']) for r in rs]}, out / 'metrics.json')
    return sc


def table_main(by, interval, dataset, view):
    steps = steps_for_minutes(interval)
    rows = []
    for name in ORDER:
        rs = by.get(name)
        if not rs: continue
        get = lambda m: m['masked_nonzero'] if view == 'masked' and 'masked_nonzero' in m else m
        ms = [get(r['metrics']) for r in rs]
        row = {'model': name, 'n_seeds': len(ms)}
        vr = np.array([val_rmse(r['dir'], r['metrics'].get('best_epoch')) if r.get('dir') else np.nan
                       for r in rs], float)
        row['VAL_RMSE (chọn mô hình)'] = fmt(np.nanmean(vr), np.nanstd(vr, ddof=1) if np.isfinite(vr).sum() > 1 else None) \
            if np.isfinite(vr).any() else '—'
        for key in ('RMSE', 'MAE', 'MAPE_percent'):
            v = np.array([m['overall'][key] for m in ms], float)
            row[f'overall_{key}'] = fmt(v.mean(), v.std(ddof=1) if len(v) > 1 else None)
        for mn in REPORT_MIN:
            k = f'{mn}min'
            for key in ('RMSE', 'MAE'):
                v = np.array([m['per_horizon'][k][key] for m in ms], float)
                row[f'{mn}min_{key}'] = fmt(v.mean(), v.std(ddof=1) if len(v) > 1 else None)
        rows.append(row)
    return pd.DataFrame(rows).set_index('model') if rows else pd.DataFrame()


def numeric_curve(by, name, view):
    rs = by.get(name)
    if not rs: return None
    get = lambda m: m['masked_nonzero'] if view == 'masked' and 'masked_nonzero' in m else m
    ms = [get(r['metrics']) for r in rs]
    ks = list(ms[0]['per_horizon'])
    r = np.array([[m['per_horizon'][k]['RMSE'] for k in ks] for m in ms])
    a = np.array([[m['per_horizon'][k]['MAE'] for k in ks] for m in ms])
    return ks, r, a


def plot_horizon(by, interval, dataset, view, names):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for i, name in enumerate(names):
        c = numeric_curve(by, name, view)
        if c is None: continue
        ks, r, a = c
        x = [(j + 1) * interval for j in range(len(ks))]
        for ax, v in zip(axes, (r, a)):
            m = v.mean(0)
            ax.plot(x, m, color=ps.color(name, i), label=ps.label(name),
                    ls='--' if name in ('naive', 'linear', 'ha_window') else '-',
                    marker='o' if len(x) <= 4 else None, ms=5)
            if len(v) > 1:
                ax.fill_between(x, v.min(0), v.max(0), color=ps.color(name, i), alpha=.15, lw=0)
    axes[0].set(xlabel='Thời gian dự báo trước (phút)', ylabel='RMSE', title='RMSE theo tầm dự báo')
    axes[1].set(xlabel='Thời gian dự báo trước (phút)', ylabel='MAE', title='MAE theo tầm dự báo')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.1))
    fig.suptitle(f'{dataset}: sai số theo từng bước (dải mờ = min–max qua 3 seed)'
                 + (' — chỉ mục tiêu ≠ 0' if view == 'masked' else ''), fontsize=11.5)
    fig.tight_layout(); fig.savefig(FIG / f'cmp_horizon_{dataset}{"_masked" if view == "masked" else ""}.png')
    plt.close(fig)


def table_paper(by, dataset, interval):
    """Paper-style (cumulative 1..k steps) metrics next to the paper's Table 1."""
    rows = []
    names = ['ha_window', 'linear', 'gcn', 'gru', 'tgcn', 'naive', 'gru_res_time', 'tgcn_res_time',
             'ensemble_gru_res_time', 'ensemble_tgcn_res_time', 'ensemble_mix']
    for name in names:
        rs = by.get(name)
        for mn in REPORT_MIN:
            k = f'{mn}min'
            row = {'model': name, 'horizon': k}
            if rs:
                ms = [r['metrics']['cumulative'][k] for r in rs]
                for key in ('RMSE', 'MAE', 'Accuracy', 'R2', 'Var'):
                    row[f'ours_{key}'] = float(np.mean([m[key] for m in ms]))
            pp = PAPER.get(dataset, {}).get(name, {}).get(mn)
            if pp:
                for key, v in zip(('RMSE', 'MAE', 'Accuracy', 'R2', 'Var'), pp):
                    row[f'paper_{key}'] = v
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(TAB / f'paper_comparison_{dataset}.csv', index=False)
    return df


def oracle_bias(runs):
    rows = []
    for r in runs:
        s = r['summary']
        if 'oracle_test_rmse' not in s or r['group'] in ('paper_protocol', 'ensemble'): continue
        rows.append({'config': f"{r['group']}/{r['name']}", 'seed': r['seed'],
                     'best_epoch_val': s['best_epoch'], 'oracle_epoch_test': s['oracle_epoch_min_test_rmse'],
                     'test_RMSE_val_selected': r['metrics']['overall']['RMSE'],
                     'test_RMSE_test_selected': s['oracle_test_rmse']})
    df = pd.DataFrame(rows)
    if len(df):
        df['optimistic_bias_%'] = 100 * (df.test_RMSE_val_selected - df.test_RMSE_test_selected) / df.test_RMSE_val_selected
    return df


def all_runs_table(runs, ds):
    """Every finished run with its per-step metrics (results/ itself is not committed)."""
    rows = []
    for r in runs:
        m = r['metrics']
        row = {'group': r['group'], 'config': r['name'], 'seed': r['seed'], 'best_epoch': m.get('best_epoch'),
               'RMSE_all': m['overall']['RMSE'], 'MAE_all': m['overall']['MAE'],
               'MAPE_all_%': m['overall']['MAPE_percent'], 'R2_all': m['overall'].get('R2')}
        for k, v in m['per_horizon'].items():
            row[f'RMSE_{k}'] = v['RMSE']; row[f'MAE_{k}'] = v['MAE']
        if 'masked_nonzero' in m:
            row['RMSE_all_nonzero'] = m['masked_nonzero']['overall']['RMSE']
            row['MAE_all_nonzero'] = m['masked_nonzero']['overall']['MAE']
        rows.append(row)
    df = pd.DataFrame(rows)
    # runs made on another machine (e.g. the GPU runs of LOCAL_RUN.md) are kept in
    # reports/external/ and merged here, so that re-running this script never drops them
    ext = ROOT / 'reports' / 'external' / f'modern_gpu_runs_{ds}.csv'
    if ext.is_file():
        e = pd.read_csv(ext, encoding='utf-8-sig')
        have = set(zip(df['group'], df['config'], df['seed'])) if len(df) else set()
        e = e[[(g, c, s_) not in have for g, c, s_ in zip(e['group'], e['config'], e['seed'])]]
        df = pd.concat([df, e], ignore_index=True)
    df.sort_values(['group', 'config', 'seed']).to_csv(TAB / f'all_runs_{ds}.csv', index=False)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--dataset', default='losloop')
    a = p.parse_args(); ds = a.dataset
    interval = 5 if ds == 'losloop' else 15
    runs, by = collect(ds)
    if not runs:
        print(f'No completed runs for {ds}.')
        return
    all_runs_table(runs, ds)
    for name in ENSEMBLE_OF:
        ens = ensemble(by, name, ds, interval)
        if ens is not None:
            by[f'ensemble_{name}'] = [{'metrics': ens, 'seed': 0, 'summary': {}, 'dir': None}]
    mx = mix_ensemble(by, ds, interval)
    if mx is not None:
        by['ensemble_mix'] = [{'metrics': mx, 'seed': 0, 'summary': {}, 'dir': None}]
    views = ['all'] + (['masked'] if ds == 'sztaxi' else [])
    md = []
    for view in views:
        df = table_main(by, interval, ds, view)
        if df.empty:
            continue
        df.to_csv(TAB / f'main_results_{ds}{"_masked" if view == "masked" else ""}.csv')
        md.append(f'### {ds} — kết quả test ({view})\n\n' + df_to_markdown(df))
        print(df.to_string())
        plot_horizon(by, interval, ds, view,
                     ['naive', 'linear', 'gcn', 'gru', 'tgcn', 'tgcn_selfsep', 'gru_res_time', 'tgcn_res_time'])
    if md:
        pdf = table_paper(by, ds, interval)
        print(pdf.round(4).to_string())
    ob = oracle_bias(runs)
    ext_ob = ROOT / 'reports' / 'external' / f'oracle_bias_modern_gpu_{ds}.csv'
    if ext_ob.is_file():
        ob = pd.concat([ob, pd.read_csv(ext_ob)], ignore_index=True).drop_duplicates(['config', 'seed'], keep='first')
    if len(ob):
        ob.to_csv(TAB / f'oracle_bias_{ds}.csv', index=False)
        print(ob.round(4).to_string())
        print('mean optimistic bias %', ob['optimistic_bias_%'].mean().round(3))
    if md:
        (TAB / f'main_results_{ds}.md').write_text('\n\n'.join(md), encoding='utf-8')


if __name__ == '__main__':
    main()
