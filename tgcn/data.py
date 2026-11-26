from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

DEFAULTS = {
    'losloop': ('2012-03-01 00:00:00', 5),
    'sztaxi': ('2015-01-01 00:00:00', 15),
    'demo': ('2026-01-01 00:00:00', 15),
    # larger public benchmarks (prepare with tools/prepare_large_dataset.py, run locally)
    'metrla': ('2012-03-01 00:00:00', 5),
    'pemsbay': ('2017-01-01 00:00:00', 5),
}

@dataclass
class DataBundle:
    values: np.ndarray  # raw [T, N]
    scaled: np.ndarray  # standardized [T, N]
    means: np.ndarray  # [N]
    stds: np.ndarray  # [N]
    adj: np.ndarray  # symmetric normalized [N,N]
    time_features: np.ndarray  # [T,4]
    splits: dict
    sensor_ids: list[str]
    interval_min: int
    data_dir: Path
    timestamps: pd.DatetimeIndex = None  # [T] wall-clock time of each step
    normalization: str = 'zscore'


def read_csv_dataset(data_dir: Path):
    path = Path(data_dir)
    speed_file, adj_file = path / 'speed.csv', path / 'adjacency.csv'
    if not speed_file.is_file() or not adj_file.is_file():
        raise FileNotFoundError(
            f'Không tìm thấy {speed_file} hoặc {adj_file}. '
            'Chạy: python download_data.py --dataset losloop (hoặc generate_demo.py).'
        )
    df = pd.read_csv(speed_file)
    if not len(df.columns):
        raise ValueError('speed.csv phải có hàng tiêu đề là sensor ID.')
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    values = df.to_numpy(dtype=np.float32)
    if np.isinf(values).any():
        values[np.isinf(values)] = np.nan
    # Interpolation uses neighboring observations, including future points. Avoid this
    # cross-split leakage: data preparation must reject NaN by default.
    if not np.isfinite(values).all():
        raise ValueError('speed.csv có NaN/Inf; hãy xử lý thiếu dữ liệu theo từng split trước khi huấn luyện.')
    adj = pd.read_csv(adj_file, header=None).to_numpy(dtype=np.float32)
    if adj.shape != (values.shape[1], values.shape[1]):
        raise ValueError(f'Adjacency shape {adj.shape} khác sensor count {values.shape[1]}.')
    if not np.isfinite(adj).all() or (adj < 0).any():
        raise ValueError('Ma trận kề chứa NaN/Inf hoặc trọng số âm.')
    if values.shape[0] < 100:
        raise ValueError('Cần tối thiểu 100 time steps.')
    return values, adj, [str(x) for x in df.columns]


def normalized_adjacency(adj: np.ndarray, mode: str = 'authors') -> np.ndarray:
    """Normalised adjacency used by the graph convolutions.

    mode='authors' (default) reproduces calculate_laplacian() of the official
    T-GCN code exactly:  a = A + I,  d = rowsum(a)^-1/2,  Â = diag(d) aᵀ diag(d).
    Note: los_adj.csv already has 1 on its diagonal, so the effective self-loop
    weight is 2; sz_adj.csv is directed and is NOT symmetrised.
    mode='symmetric_selfloop1': max(A, Aᵀ) with the diagonal set to 1 (the
    first version of this project; kept for the self-loop ablation).
    """
    a = np.asarray(adj, dtype=np.float64)
    if mode == 'symmetric_selfloop1':
        a = np.maximum(a, a.T).copy(); np.fill_diagonal(a, 1.0)
        d = 1.0 / np.sqrt(np.maximum(a.sum(1), 1e-6))
        return (d[:, None] * a * d[None, :]).astype(np.float32)
    a = a + np.eye(len(a))
    d = np.power(a.sum(1), -0.5); d[~np.isfinite(d)] = 0.0
    return (d[:, None] * a.T * d[None, :]).astype(np.float32)


def make_time_features(length: int, start_time: str, interval_min: int) -> np.ndarray:
    dates = pd.date_range(start=start_time, periods=length, freq=f'{interval_min}min')
    tod = (dates.hour.to_numpy() * 60 + dates.minute.to_numpy()) / (24 * 60)
    dow = dates.dayofweek.to_numpy() / 7.0
    t = np.stack([np.sin(2*np.pi*tod), np.cos(2*np.pi*tod),
                  np.sin(2*np.pi*dow), np.cos(2*np.pi*dow)], axis=-1)
    return t.astype(np.float32)


def load_bundle(data_dir: str | Path, dataset: str, seq_len=12, horizon=3,
                train_ratio=0.7, val_ratio=0.1, interval_min=None, start_time=None,
                normalization='zscore', protocol='strict') -> DataBundle:
    """Chronological split train | val | test.

    normalization='zscore': per-sensor mean/std of the TRAIN part (default).
    normalization='max'   : divide everything by the global max of the TRAIN part
                            (the authors divide by the max of the whole series,
                            which leaks a test statistic; we keep it train-only).
    """
    values, adj, sensor_ids = read_csv_dataset(Path(data_dir))
    if protocol == 'paper':
        return _paper_bundle(values, adj, sensor_ids, dataset, seq_len, horizon,
                             interval_min, start_time, data_dir)
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError('0 < train_ratio, val_ratio và train_ratio+val_ratio < 1.')
    t = len(values)
    train_end, val_end = int(t*train_ratio), int(t*(train_ratio+val_ratio))
    if min(train_end - seq_len - horizon, val_end - train_end - horizon,
           t - val_end - horizon) <= 0:
        raise ValueError('Split quá ngắn so với seq_len, horizon.')
    if normalization == 'zscore':
        means = values[:train_end].mean(axis=0)
        stds = values[:train_end].std(axis=0)
        stds[stds < 1e-6] = 1.0
    elif normalization == 'max':
        means = np.zeros(values.shape[1], dtype=np.float32)
        stds = np.full(values.shape[1], float(values[:train_end].max()), dtype=np.float32)
    else:
        raise ValueError(f'Unknown normalization {normalization}')
    scaled = (values - means[None, :]) / stds[None, :]
    default_start, default_interval = DEFAULTS.get(dataset, DEFAULTS['demo'])
    interval_min = default_interval if interval_min is None else interval_min
    start_time = default_start if start_time is None else start_time
    return DataBundle(values, scaled.astype(np.float32), means.astype(np.float32),
                      stds.astype(np.float32),
                      normalized_adjacency(adj),
                      make_time_features(t, start_time, interval_min),
                      {'train': (seq_len, train_end),
                       'val': (train_end, val_end),
                       'test': (val_end, t)},
                      sensor_ids, interval_min, Path(data_dir),
                      pd.date_range(start=start_time, periods=t, freq=f'{interval_min}min'),
                      normalization)


def _paper_bundle(values, adj, sensor_ids, dataset, seq_len, horizon, interval_min, start_time, data_dir):
    """Replicates the authors' protocol (input_data.py / main.py of lehaifeng/T-GCN):
    80% train / 20% test, x / max(whole series), test windows built only from the
    test segment, and *the test set doubles as validation* (they keep the epoch
    with the minimum test RMSE). Used only to reproduce and criticise Table 1."""
    t = len(values); train_end = int(t * 0.8)
    mx = float(values.max())
    means = np.zeros(values.shape[1], dtype=np.float32)
    stds = np.full(values.shape[1], mx, dtype=np.float32)
    default_start, default_interval = DEFAULTS.get(dataset, DEFAULTS['demo'])
    interval_min = default_interval if interval_min is None else interval_min
    start_time = default_start if start_time is None else start_time
    test = (train_end + seq_len, t)
    return DataBundle(values, (values / mx).astype(np.float32), means, stds,
                      normalized_adjacency(adj), make_time_features(t, start_time, interval_min),
                      {'train': (seq_len, train_end), 'val': test, 'test': test},
                      sensor_ids, interval_min, Path(data_dir),
                      pd.date_range(start=start_time, periods=t, freq=f'{interval_min}min'), 'max')


class TrafficWindows(Dataset):
    """Targets within split; val/test past context can cross earlier split boundary."""
    def __init__(self, bundle: DataBundle, split: str, seq_len: int, horizon: int):
        self.data = bundle.scaled
        self.tfeat = bundle.time_features
        self.seq_len = seq_len
        self.horizon = horizon
        left, right = bundle.splits[split]
        self.starts = np.arange(left, right - horizon + 1, dtype=np.int64)
        if len(self.starts) <= 0:
            raise ValueError(f'Không đủ time steps cho {split}.')

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, index):
        t = int(self.starts[index]); l = self.seq_len; h = self.horizon
        return (torch.from_numpy(self.data[t-l:t].copy()).unsqueeze(-1),
                torch.from_numpy(self.tfeat[t-l:t].copy()),
                torch.from_numpy(self.data[t:t+h].copy()),
                torch.tensor(t, dtype=torch.long))


def make_loaders(bundle, seq_len, horizon, batch_size, workers=0, eval_batch_size=256):
    sets = {s: TrafficWindows(bundle, s, seq_len, horizon) for s in ('train','val','test')}
    loaders = {s: DataLoader(ds, batch_size=(batch_size if s == 'train' else eval_batch_size),
                             shuffle=(s=='train'),
                             num_workers=workers, pin_memory=torch.cuda.is_available())
               for s, ds in sets.items()}
    return loaders


def target_mask(bundle: DataBundle, t_index: torch.Tensor, horizon: int) -> torch.Tensor:
    """Boolean [B,H,N]: True where the raw target is observed (non-zero)."""
    rows = t_index[:, None] + torch.arange(horizon)[None, :]
    return torch.from_numpy(bundle.values != 0)[rows]


def inverse_transform(x: np.ndarray, bundle: DataBundle) -> np.ndarray:
    return x * bundle.stds[None, None, :] + bundle.means[None, None, :]


def tod_profile_z(bundle, smooth=2):
    """Per-sensor mean z-score of every time-of-day slot over the TRAIN part only,
    circularly smoothed over +-smooth slots. Shape [steps_per_day, N]."""
    slots = int(round(1440 / bundle.interval_min))
    ts = bundle.timestamps
    slot = ((ts.hour * 60 + ts.minute) // bundle.interval_min).to_numpy()
    tr = bundle.splits['train'][1]
    z = bundle.scaled[:tr].astype(np.float64)
    prof = np.zeros((slots, z.shape[1]))
    for k in range(slots):
        sel = slot[:tr] == k
        if sel.any():
            prof[k] = z[sel].mean(0)
    kern = np.ones(2 * smooth + 1) / (2 * smooth + 1)
    prof = np.stack([np.convolve(np.r_[prof[-smooth:, j], prof[:, j], prof[:smooth, j]], kern, 'valid')
                     for j in range(prof.shape[1])], axis=1)
    return prof.astype(np.float32)
