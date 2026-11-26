"""Convert the standard METR-LA / PEMS-BAY files to this project's format so that
every model and analysis script runs on them unchanged (run on your own machine).

Sources (public, from the DCRNN authors, Li et al. 2018 — 2nd paper in Papers.xlsx):
  speeds   : metr-la.h5 / pems-bay.h5    (pandas HDF5: index = timestamps, columns = sensor ids)
             or a CSV with the same layout (e.g. PEMS-BAY.csv from zenodo.org/records/5146275)
  graph    : adj_mx.pkl / adj_mx_bay.pkl (pickle: sensor_ids, sensor_id_to_ind, adj_mx)
             https://github.com/liyaguang/DCRNN  -> README "Data Preparation" links

    pip install tables                 # needed to read .h5
    python tools/prepare_large_dataset.py --name metrla  --speeds metr-la.h5  --adj adj_mx.pkl
    python tools/prepare_large_dataset.py --name pemsbay --speeds pems-bay.h5 --adj adj_mx_bay.pkl

Writes data/<name>/speed.csv, data/<name>/adjacency.csv and data/<name>/meta.json.
Zeros in METR-LA are missing readings: train/evaluate with --mask-zero (metrics on
non-zero targets are reported automatically as 'masked_nonzero').
"""
import argparse
import json
import pickle
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--name', required=True, choices=['metrla', 'pemsbay'])
    ap.add_argument('--speeds', required=True, help='.h5 or .csv (timestamps x sensors)')
    ap.add_argument('--adj', required=True, help='DCRNN adj_mx*.pkl')
    a = ap.parse_args()
    sp = Path(a.speeds)
    df = pd.read_hdf(sp) if sp.suffix in ('.h5', '.hdf5') else pd.read_csv(sp, index_col=0, parse_dates=True)
    with open(a.adj, 'rb') as f:
        try:
            ids, id2ind, adj = pickle.load(f)
        except UnicodeDecodeError:
            f.seek(0); ids, id2ind, adj = pickle.load(f, encoding='latin1')
    ids = [str(x) for x in ids]
    df.columns = [str(c) for c in df.columns]
    missing = [i for i in ids if i not in df.columns]
    if missing:
        sys.exit(f'{len(missing)} sensors of the graph are not in the speed file')
    df = df[ids].astype(np.float32)
    if df.isna().any().any():
        df = df.fillna(0.0)          # treated as missing (zero) like the original benchmark
    adj = np.asarray(adj, dtype=np.float32)
    out = ROOT / 'data' / a.name; out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / 'speed.csv', index=False)
    pd.DataFrame(adj).to_csv(out / 'adjacency.csv', header=False, index=False)
    idx = pd.DatetimeIndex(df.index) if not isinstance(df.index, pd.RangeIndex) else None
    meta = {'rows': int(len(df)), 'sensors': len(ids), 'zero_share': float((df.values == 0).mean()),
            'start_time': str(idx[0]) if idx is not None else None,
            'interval_min': int((idx[1] - idx[0]).total_seconds() // 60) if idx is not None else 5}
    (out / 'meta.json').write_text(json.dumps(meta, indent=2))
    print('wrote', out, meta)
    if idx is not None:
        print(f"train.py: --dataset {a.name} --start-time '{meta['start_time']}' --interval-min {meta['interval_min']}")


if __name__ == '__main__':
    main()
