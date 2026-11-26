"""Helpers shared by the analysis scripts: locate runs, load predictions/models."""
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import argparse
import numpy as np
import pandas as pd
import torch
from tgcn.data import load_bundle, make_loaders
from tgcn.utility import load_checkpoint

FIG = ROOT / 'reports' / 'figures'; FIG.mkdir(parents=True, exist_ok=True)
TAB = ROOT / 'reports' / 'tables'; TAB.mkdir(parents=True, exist_ok=True)
H_DEFAULT = {'losloop': 12, 'sztaxi': 4, 'metrla': 12, 'pemsbay': 12}
# horizons reported (minutes) and their step index
REPORT_MIN = [15, 30, 45, 60]

# Zhao et al. (2019), Table 1 (cumulative over the first k steps, test-selected epoch)
PAPER = {
    'losloop': {
        'ha_window': {15: (7.4427, 4.0145, .8733, .7121, .7121), 30: (7.4427, 4.0145, .8733, .7121, .7121),
                      45: (7.4427, 4.0145, .8733, .7121, .7121), 60: (7.4427, 4.0145, .8733, .7121, .7121)},
        'linear': {15: (6.0084, 3.7285, .8977, .8123, .8146), 30: (6.9588, 3.7248, .8815, .7492, .7523),
                   45: (7.7504, 4.1288, .8680, .6899, .6947), 60: (8.4388, 4.5036, .8562, .6336, .5593)},
        'gcn': {15: (7.7922, 5.3525, .8673, .6843, .6844), 30: (8.3353, 5.6118, .8581, .6402, .6404),
                45: (8.8036, 5.9534, .8500, .5999, .6001), 60: (9.2657, 6.2892, .8421, .5583, .5593)},
        'gru': {15: (5.2182, 3.0602, .9109, .8576, .8577), 30: (6.2802, 3.6505, .8931, .7957, .7958),
                45: (7.0343, 4.0915, .8801, .7446, .7451), 60: (7.6621, 4.5186, .8694, .6980, .6984)},
        'tgcn': {15: (5.1264, 3.1802, .9127, .8634, .8634), 30: (6.0598, 3.7466, .8968, .8098, .8100),
                 45: (6.7065, 4.1158, .8857, .7679, .7684), 60: (7.2677, 4.6021, .8762, .7283, .7290)},
    },
    'sztaxi': {
        'ha_window': {k: (7.9198, 5.4969, .6807, .7914, .7914) for k in REPORT_MIN},
        'linear': {15: (7.5368, 4.9269, .6961, .8111, .8121), 30: (7.4747, 4.9819, .6987, .8142, .8144),
                   45: (7.4755, 5.0332, .6986, .8141, .8142), 60: (7.4883, 5.0714, .6981, .8135, .8136)},
        'gcn': {15: (9.2717, 7.2606, .6433, .6147, .6147), 30: (9.3450, 7.3211, .6405, .6086, .6086),
                45: (9.4023, 7.3704, .6383, .6038, .6039), 60: (9.4504, 7.4120, .6365, .5998, .5999)},
        'gru': {15: (4.0483, 2.6814, .7178, .8498, .8499), 30: (4.0769, 2.7009, .7158, .8477, .8477),
                45: (4.1002, 2.7207, .7142, .8460, .8459), 60: (4.1241, 2.7431, .7125, .8442, .8321)},
        'tgcn': {15: (3.9162, 2.7061, .7306, .8541, .8626), 30: (3.9617, 2.7452, .7275, .8523, .8523),
                 45: (3.9950, 2.7666, .7252, .8509, .8509), 60: (4.0141, 2.7889, .7238, .8503, .8504)},
    },
}


def run_dirs(dataset):
    base = ROOT / 'results' / dataset
    out = []
    for f in sorted(base.glob('**/metrics.json')):
        rel = f.parent.relative_to(base).parts
        m = json.loads(f.read_text())
        if m.get('caution'):
            continue
        group = rel[0]
        name = rel[1] if len(rel) > 1 else rel[0]
        seed = m.get('seed', 0)
        summ = {}
        if (f.parent / 'training_summary.json').is_file():
            summ = json.loads((f.parent / 'training_summary.json').read_text())
        out.append({'group': group, 'name': name, 'seed': seed, 'dir': f.parent,
                    'metrics': m, 'summary': summ})
    return out


def val_rmse(run_dir, epoch=None):
    """Validation RMSE in original units at the selected (best-validation) epoch.
    Comparable across loss functions -> used to choose between configurations."""
    h = Path(run_dir) / 'history.csv'
    if not h.is_file():
        return float('nan')
    df = pd.read_csv(h)
    if 'val_rmse_raw' not in df:
        return float('nan')
    if epoch is None:
        epoch = int(df.epoch[df.val_mse_scaled.idxmin()])
    return float(df.loc[df.epoch == epoch, 'val_rmse_raw'].iloc[0])


def load_pred(run_dir):
    z = np.load(Path(run_dir) / 'test_predictions.npz')
    return z['y_true'], z['y_pred'], z['time_indices']


def bundle_for(dataset, seq_len=12, horizon=None):
    return load_bundle(ROOT / 'data' / dataset, dataset, seq_len, horizon or H_DEFAULT[dataset])


def load_model(run_dir, device='cpu'):
    from train import build_model
    ck = load_checkpoint(Path(run_dir) / 'best.pt', device)
    cfg = ck['config']
    b = load_bundle(ROOT / 'data' / cfg['dataset'], cfg['dataset'], cfg['seq_len'], cfg['horizon'],
                    cfg['train_ratio'], cfg['val_ratio'], cfg['interval_min'], cfg['start_time'],
                    cfg.get('normalization', 'zscore'))
    model = build_model(argparse.Namespace(**cfg), b).to(device)
    model.load_state_dict(ck['state_dict']); model.eval()
    return model, b, cfg


def steps_for_minutes(interval, minutes=REPORT_MIN):
    return [m // interval - 1 for m in minutes]


def fmt(mean, std=None, d=3):
    if std is None or not np.isfinite(std):
        return f'{mean:.{d}f}'
    return f'{mean:.{d}f} ± {std:.{d}f}'


def df_to_markdown(df, index=True):
    cols = ([df.index.name or ''] if index else []) + [str(c) for c in df.columns]
    lines = ['| ' + ' | '.join(cols) + ' |', '|' + '|'.join(['---'] * len(cols)) + '|']
    for idx, row in df.iterrows():
        vals = ([str(idx)] if index else []) + [str(v) for v in row.values]
        lines.append('| ' + ' | '.join(vals) + ' |')
    return '\n'.join(lines)
