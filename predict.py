"""Load a trained checkpoint and forecast the next H steps for every sensor.

What a checkpoint (best.pt) contains
    state_dict  : the learned weights (GCN-GRU gates, candidate, output head, ...)
    config      : every hyper-parameter used by train.py (model, seq_len, horizon,
                  hidden_dim, residual, time_features, ...)
    data_stats  : per-sensor mean/std of the training split, the normalized
                  adjacency, sensor ids, sampling interval -> the checkpoint is
                  self-contained (older checkpoints without it are re-derived
                  from the training CSV).

Usage
    # forecast after the last row of the CSV
    python predict.py --checkpoint models/losloop_gru_res_time_seed42.pt --input data/losloop/speed.csv
    # final model = ensemble of 3 seeds, forecast from a given moment, compare with truth
    python predict.py --checkpoint models/losloop_gru_res_time_seed*.pt \\
        --input data/losloop/speed.csv --end-time "2012-03-07 07:30" --compare
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tgcn.data import load_bundle, make_time_features
from tgcn.utility import ROOT, load_checkpoint
from train import build_model


class Forecaster:
    """Thin wrapper: raw speeds [L, N] (+ their timestamps) -> raw forecast [H, N]."""

    def __init__(self, checkpoint, device='cpu', data_dir=None):
        ck = load_checkpoint(checkpoint, device)
        self.cfg = cfg = ck['config']
        stats = ck.get('data_stats')
        if stats is None:  # older checkpoint: recompute train statistics
            b = load_bundle(data_dir or ROOT / 'data' / cfg['dataset'], cfg['dataset'],
                            cfg['seq_len'], cfg['horizon'], cfg['train_ratio'], cfg['val_ratio'],
                            cfg['interval_min'], cfg['start_time'], cfg.get('normalization', 'zscore'))
            stats = {'means': torch.as_tensor(b.means), 'stds': torch.as_tensor(b.stds),
                     'adj_normalized': torch.as_tensor(b.adj), 'sensor_ids': b.sensor_ids,
                     'interval_min': b.interval_min, 'start_time': str(b.timestamps[0])}
        self.means = stats['means'].numpy(); self.stds = stats['stds'].numpy()
        self.sensor_ids = list(stats['sensor_ids']); self.interval = int(stats['interval_min'])
        self.start_time = stats['start_time']

        class _B:  # minimal bundle for build_model
            adj = stats['adj_normalized'].numpy(); sensor_ids = self.sensor_ids
            interval_min = int(stats['interval_min'])
        self.model = build_model(argparse.Namespace(**cfg), _B).to(device)
        self.model.load_state_dict(ck['state_dict']); self.model.eval()
        self.device = device; self.best_epoch = ck.get('best_epoch')

    @property
    def seq_len(self): return int(self.cfg['seq_len'])

    @property
    def horizon(self): return int(self.cfg['horizon'])

    def n_parameters(self):
        return int(sum(p.numel() for p in self.model.parameters()))

    @torch.no_grad()
    def predict(self, window_raw, window_times):
        """window_raw [L,N] speeds; window_times: L timestamps. Returns [H,N] speeds."""
        L = self.seq_len
        w = np.asarray(window_raw, dtype=np.float32)
        assert w.shape == (L, len(self.sensor_ids)), f'need window shape {(L, len(self.sensor_ids))}'
        x = (w - self.means) / self.stds
        tf = make_time_features(L, str(pd.Timestamp(window_times[0])), self.interval)
        x = torch.from_numpy(x)[None, :, :, None].to(self.device)
        tf = torch.from_numpy(tf)[None].to(self.device)
        out = self.model(x, tf)[0].cpu().numpy()            # [H,N] standardized
        return out * self.stds + self.means


def predict_gbdt(a):
    from gbdt import GBDTForecaster
    g = GBDTForecaster(a.checkpoint[0])
    df = pd.read_csv(a.input); df.columns = [str(c) for c in df.columns]
    times = pd.date_range(a.start_time or str(g.st['start_time']), periods=len(df), freq=f'{g.interval}min')
    end = len(df) - 1 if a.end_time is None else int(times.get_indexer([pd.Timestamp(a.end_time)])[0])
    fc, _ = g.predict(df, end, a.start_time)
    target_times = pd.date_range(times[end] + pd.Timedelta(minutes=g.interval), periods=g.H, freq=f'{g.interval}min')
    out = pd.DataFrame(fc, index=target_times, columns=g.sensor_ids); out.index.name = 'target_time'
    print(f'LightGBM ({len(g.boosters)} boosters, one per horizon, features v{g.version}) | input ends {times[end]}')
    show = out.iloc[:, :a.sensors].round(1)
    if a.compare and end + g.H < len(df):
        act = df.iloc[end + 1:end + 1 + g.H, :a.sensors].to_numpy()
        show = pd.concat({'forecast': show, 'actual': pd.DataFrame(act, index=out.index, columns=show.columns)}, axis=1)
    print(show.to_string())
    if a.output:
        out.to_csv(a.output); print('Saved', a.output)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', required=True, nargs='+',
                   help='one checkpoint, or several (same dataset) to average = ensemble')
    p.add_argument('--input', required=True, help='speed CSV: header = sensor ids, one row per step')
    p.add_argument('--start-time', default=None, help='timestamp of the first CSV row')
    p.add_argument('--end-time', default=None, help='last observed moment (default: last row)')
    p.add_argument('--output', default=None, help='CSV to write [H x N] forecast')
    p.add_argument('--compare', action='store_true', help='print actual values if they exist in the CSV')
    p.add_argument('--sensors', type=int, default=5, help='how many sensors to print')
    a = p.parse_args()
    if len(a.checkpoint) == 1 and Path(a.checkpoint[0]).is_dir():      # LightGBM model directory
        return predict_gbdt(a)
    members = [Forecaster(c) for c in a.checkpoint]
    f = members[0]
    if any(m.seq_len != f.seq_len or m.horizon != f.horizon or m.sensor_ids != f.sensor_ids for m in members):
        raise SystemExit('Ensemble members must share sensors, seq_len and horizon.')
    df = pd.read_csv(a.input)
    df.columns = [str(c) for c in df.columns]
    if list(df.columns) != f.sensor_ids:
        raise SystemExit('Sensor columns of the CSV differ from those the model was trained on.')
    times = pd.date_range(a.start_time or f.start_time, periods=len(df), freq=f'{f.interval}min')
    end = len(df) - 1 if a.end_time is None else int(times.get_indexer([pd.Timestamp(a.end_time)])[0])
    if end < f.seq_len - 1:
        raise SystemExit('Not enough history before --end-time.')
    win = df.iloc[end - f.seq_len + 1:end + 1].to_numpy(np.float32)
    fc = np.mean([m.predict(win, times[end - f.seq_len + 1:end + 1]) for m in members], axis=0)
    fc = np.clip(fc, 0.0, None)          # speeds cannot be negative (T-GCN can output < 0)
    target_times = pd.date_range(times[end] + pd.Timedelta(minutes=f.interval), periods=f.horizon,
                                 freq=f'{f.interval}min')
    out = pd.DataFrame(fc, index=target_times, columns=f.sensor_ids)
    out.index.name = 'target_time'
    print(f"Model {f.cfg['model']} (residual={f.cfg.get('residual')}, time_features={f.cfg.get('time_features')}, "
          f"self_sep={f.cfg.get('self_sep', False)}) | {f.n_parameters()} parameters"
          + (f' | ensemble of {len(members)} checkpoints' if len(members) > 1 else f' | best epoch {f.best_epoch}'))
    print(f'Input: {f.seq_len} steps ending {times[end]}; forecast {f.horizon} steps x {f.interval} min')
    show = out.iloc[:, :a.sensors].round(1)
    if a.compare and end + f.horizon < len(df):
        act = df.iloc[end + 1:end + 1 + f.horizon, :a.sensors].to_numpy()
        show = pd.concat({'forecast': show, 'actual': pd.DataFrame(act, index=out.index,
                                                                    columns=show.columns)}, axis=1)
    print(show.to_string())
    if a.output:
        out.to_csv(a.output); print('Saved', a.output)


if __name__ == '__main__':
    main()
