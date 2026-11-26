"""Every model against the original baseline (T-GCN), in one table per dataset.

Two ways of scoring, two baselines:
  * strict protocol (ours: 70/10/20, epoch chosen on validation, 3 seeds), errors at each
    step -> baseline = our reproduction of the original T-GCN (mean of 3 seeds);
  * the paper's way ("15/30/45/60 min" = errors pooled over steps 1..k) -> baseline =
    T-GCN as printed in Table 1 of Zhao et al. (2019).
Runs made on the user's GPU (reports/external/) enter through their per-step errors; their
pooled metrics are derived exactly by analysis/gpu_results.py.

    python analysis/gpu_results.py && python analysis/baseline_compare.py
"""
import json
import numpy as np
import pandas as pd
from common import ROOT, TAB, PAPER, run_dirs, val_rmse

EXT = ROOT / 'reports' / 'external'
STEP_MIN = {'losloop': (5, 15, 30, 60), 'sztaxi': (15, 30, 45, 60)}
CUM_MIN = (15, 30, 45, 60)

# key, label, kind, source, shown in the summary tables at the top of the report
ENTRIES = {
    'losloop': [
        ('paper_tgcn', 'T-GCN — số trong bài báo', 'paper', 'tgcn', 1),
        ('paper_gru', 'GRU — số trong bài báo', 'paper', 'gru', 1),
        ('naive', 'Persistence (giữ giá trị cuối)', 'local', (('baselines',), 'naive'), 1),
        ('linear', 'Hồi quy tuyến tính', 'local', (('baselines',), 'linear'), 0),
        ('gcn', 'GCN', 'local', (('main',), 'gcn'), 0),
        ('gru', 'GRU', 'local', (('main',), 'gru'), 1),
        ('tgcn', 'T-GCN gốc (tái lập)', 'local', (('main',), 'tgcn'), 1),
        ('tgcn_selfsep', 'T-GCN tách self/láng giềng', 'local', (('improve', 'variants'), 'tgcn_selfsep'), 1),
        ('gru_res_time', 'GRU-RT (residual + giờ)', 'local', (('improve',), 'gru_res_time'), 1),
        ('ens_gru_res_time', 'Ensemble GRU-RT (3 seed)', 'ens', 'gru_res_time', 1),
        ('stid', 'STID', 'local', (('modern',), 'stid'), 1),
        ('dcrnn_res', 'DCRNN-lite + residual (1 seed)', 'local', (('modern',), 'dcrnn_res'), 1),
        ('gwnet_res', 'Graph WaveNet + residual (CPU)', 'local', (('modern',), 'gwnet_res'), 1),
        ('gpu_gwnet_res', 'Graph WaveNet + residual (GPU)', 'gpu', 'gwnet_res', 1),
        ('gpu_gwnet_full_res', 'Graph WaveNet + residual, cỡ đầy đủ (GPU)', 'gpu', 'gwnet_full_res', 1),
        ('gpu_staeformer_full', 'STAEformer cỡ đầy đủ (GPU)', 'gpu', 'staeformer_full', 1),
        ('lgbm_best', 'LightGBM v2 (đặc trưng KG–TG)', 'local', (('modern',), 'lgbm_best'), 1),
        ('blend', 'Trộn theo tầm: LightGBM + Graph WaveNet-res (+ GRU-RT) — mô hình cuối', 'ens', 'blend', 1),
    ],
    'sztaxi': [
        ('paper_tgcn', 'T-GCN — số trong bài báo', 'paper', 'tgcn', 1),
        ('paper_gru', 'GRU — số trong bài báo', 'paper', 'gru', 1),
        ('naive', 'Persistence (giữ giá trị cuối)', 'local', (('baselines',), 'naive'), 1),
        ('ha_window', 'HA (TB 12 bước đầu vào)', 'local', (('baselines',), 'ha_window'), 1),
        ('gru', 'GRU', 'local', (('main',), 'gru'), 1),
        ('tgcn', 'T-GCN gốc (tái lập)', 'local', (('main',), 'tgcn'), 1),
        ('gru_rtp', 'GRU-RTP (+ hồ sơ giờ)', 'local', (('modern',), 'gru_rtp'), 0),
        ('stid', 'STID (CPU)', 'local', (('modern',), 'stid'), 1),
        ('gpu_gwnet', 'Graph WaveNet (GPU)', 'gpu', 'gwnet', 1),
        ('gpu_gwnet_full', 'Graph WaveNet cỡ đầy đủ (GPU)', 'gpu', 'gwnet_full', 1),
        ('gpu_staeformer', 'STAEformer (GPU)', 'gpu', 'staeformer', 1),
        ('lgbm_best', 'LightGBM v2 (đặc trưng KG–TG)', 'local', (('modern',), 'lgbm_best'), 1),
        ('blend', 'Trộn theo tầm: STID-res + LightGBM (+ GRU-RTP, GRU) — mô hình cuối', 'ens', 'blend', 1),
    ],
}


def agg(rows):
    """mean (and sample sd if > 1 run) of every metric."""
    d = pd.DataFrame(rows)
    out = d.mean(numeric_only=True).to_dict()
    for c in ('RMSE', 'MAE', 'RMSE_nz', 'MAE_nz'):
        if c in d:
            out[c + '_sd'] = d[c].std(ddof=1) if len(d) > 1 else np.nan
    out['n'] = len(d)
    return out


def from_metrics(m, ds):
    r = {'RMSE': m['overall']['RMSE'], 'MAE': m['overall']['MAE']}
    for k in STEP_MIN[ds]:
        r[f's{k}_RMSE'] = m['per_horizon'][f'{k}min']['RMSE']; r[f's{k}_MAE'] = m['per_horizon'][f'{k}min']['MAE']
    for k in CUM_MIN:
        c = m['cumulative'][f'{k}min']
        r.update({f'c{k}_RMSE': c['RMSE'], f'c{k}_MAE': c['MAE'], f'c{k}_Acc': c['Accuracy'], f'c{k}_R2': c['R2']})
    if 'masked_nonzero' in m:
        z = m['masked_nonzero']
        r.update({'RMSE_nz': z['overall']['RMSE'], 'MAE_nz': z['overall']['MAE']})
        for k in STEP_MIN[ds]:
            r[f'nz{k}_RMSE'] = z['per_horizon'][f'{k}min']['RMSE']; r[f'nz{k}_MAE'] = z['per_horizon'][f'{k}min']['MAE']
    return r


def local(ds, groups, name):
    rs = [r for r in run_dirs(ds) if r['group'] in groups and r['name'] == name]
    if not rs:
        return None
    rows = []
    for r in rs:
        x = from_metrics(r['metrics'], ds)
        if name.startswith('lgbm'):
            x['val_RMSE'] = json.loads((r['dir'] / 'training_summary.json').read_text())['val_rmse_raw']
        elif r['group'] != 'baselines':
            x['val_RMSE'] = val_rmse(r['dir'], r['metrics'].get('best_epoch'))
        rows.append(x)
    return agg(rows)


def ensemble(ds, tag):
    f = ROOT / 'results' / ds / 'ensemble' / tag / 'metrics.json'
    if not f.is_file():
        return None
    x = from_metrics(json.loads(f.read_text()), ds)
    x['n'] = 1
    return x


def gpu(ds, cfg):
    f = EXT / f'modern_gpu_runs_{ds}.csv'
    if not f.is_file():
        return None
    runs = pd.read_csv(f, encoding='utf-8-sig')
    cum = pd.read_csv(TAB / f'modern_gpu_cumulative_{ds}.csv')
    runs = runs[runs.config == cfg].merge(cum, on=['config', 'seed'])
    rows = []
    for _, r in runs.iterrows():
        x = {'RMSE': r.RMSE_all, 'MAE': r.MAE_all}
        for k in STEP_MIN[ds]:
            x[f's{k}_RMSE'] = r[f'RMSE_{k}min']; x[f's{k}_MAE'] = r[f'MAE_{k}min']
        for k in CUM_MIN:
            x.update({f'c{k}_RMSE': r[f'RMSE_cum{k}'], f'c{k}_MAE': r[f'MAE_cum{k}'],
                      f'c{k}_Acc': r[f'Accuracy_cum{k}'], f'c{k}_R2': r[f'R2_cum{k}']})
        if 'RMSE_all_nonzero' in r:
            x.update({'RMSE_nz': r.RMSE_all_nonzero, 'MAE_nz': r.MAE_all_nonzero})
        rows.append(x)
    return agg(rows) if rows else None


def paper(ds, key):
    x = {'n': np.nan}
    for k in CUM_MIN:
        rm, ma, acc, r2, _ = PAPER[ds][key][k]
        x.update({f'c{k}_RMSE': rm, f'c{k}_MAE': ma, f'c{k}_Acc': acc, f'c{k}_R2': r2})
    return x


def main():
    for ds, entries in ENTRIES.items():
        rows = []
        for key, label, kind, src, summ in entries:
            x = {'paper': lambda: paper(ds, src), 'local': lambda: local(ds, *src),
                 'ens': lambda: ensemble(ds, src), 'gpu': lambda: gpu(ds, src)}[kind]()
            if x is None:
                print('missing', ds, key); continue
            rows.append({'key': key, 'label': label, 'kind': kind, 'summary': summ, **x})
        df = pd.DataFrame(rows)
        if df.empty or not {'tgcn', 'paper_tgcn'} <= set(df.key):
            print(f'{ds}: the T-GCN baseline runs are missing (results/ is not in git) - table left unchanged')
            continue
        base = df.set_index('key')
        for c in ('RMSE', 'MAE', 'RMSE_nz', 'MAE_nz'):
            if c in df:
                df[f'd_{c}_%'] = 100 * (df[c] / base.loc['tgcn', c] - 1)
        for c in ('c15_RMSE', 'c15_MAE', 'c60_RMSE', 'c60_MAE'):
            df[f'dp_{c}_%'] = 100 * (df[c] / base.loc['paper_tgcn', c] - 1)
        df.to_csv(TAB / f'vs_baseline_{ds}.csv', index=False)
        show = ['label', 'n', 'RMSE', 'MAE', 'd_RMSE_%', 'd_MAE_%', 'c15_RMSE', 'c60_RMSE', 'dp_c60_RMSE_%']
        print(f'== {ds}\n', df[[c for c in show if c in df]].round(3).to_string(index=False))


if __name__ == '__main__':
    main()
