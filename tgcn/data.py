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


def normalized_adjacency(adj: np.ndarray) -> np.ndarray:
    a = np.maximum(adj, adj.T).astype(np.float32, copy=True)
    np.fill_diagonal(a, 1.0)
    degrees = np.sum(a, axis=1)
    inv = 1.0 / np.sqrt(np.maximum(degrees, 1e-6))
    norm = inv[:, None] * a * inv[None, :]
    return norm.astype(np.float32)


def make_time_features(length: int, start_time: str, interval_min: int) -> np.ndarray:
    dates = pd.date_range(start=start_time, periods=length, freq=f'{interval_min}min')
    tod = (dates.hour.to_numpy() * 60 + dates.minute.to_numpy()) / (24 * 60)
    dow = dates.dayofweek.to_numpy() / 7.0
    t = np.stack([np.sin(2*np.pi*tod), np.cos(2*np.pi*tod),
                  np.sin(2*np.pi*dow), np.cos(2*np.pi*dow)], axis=-1)
    return t.astype(np.float32)


def load_bundle(data_dir: str | Path, dataset: str, seq_len=12, horizon=3,
                train_ratio=0.7, val_ratio=0.1, interval_min=None, start_time=None) -> DataBundle:
    values, adj, sensor_ids = read_csv_dataset(Path(data_dir))
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError('0 < train_ratio, val_ratio và train_ratio+val_ratio < 1.')
    t = len(values)
    train_end, val_end = int(t*train_ratio), int(t*(train_ratio+val_ratio))
    if min(train_end - seq_len - horizon, val_end - train_end - horizon,
           t - val_end - horizon) <= 0:
        raise ValueError('Split quá ngắn so với seq_len, horizon.')
    means = values[:train_end].mean(axis=0)
    stds = values[:train_end].std(axis=0)
    stds[stds < 1e-6] = 1.0
    scaled = (values - means[None, :]) / stds[None, :]
    default_start, default_interval = DEFAULTS.get(dataset, DEFAULTS['demo'])
    interval_min = default_interval if interval_min is None else interval_min
    start_time = default_start if start_time is None else start_time
    return DataBundle(values, scaled.astype(np.float32), means, stds,
                      normalized_adjacency(adj),
                      make_time_features(t, start_time, interval_min),
                      {'train': (seq_len, train_end),
                       'val': (train_end, val_end),
                       'test': (val_end, t)},
                      sensor_ids, interval_min, Path(data_dir))


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


def make_loaders(bundle, seq_len, horizon, batch_size, workers=0):
    sets = {s: TrafficWindows(bundle, s, seq_len, horizon) for s in ('train','val','test')}
    loaders = {s: DataLoader(ds, batch_size=batch_size, shuffle=(s=='train'),
                             num_workers=workers, pin_memory=torch.cuda.is_available())
               for s, ds in sets.items()}
    return loaders


def inverse_transform(x: np.ndarray, bundle: DataBundle) -> np.ndarray:
    return x * bundle.stds[None, None, :] + bundle.means[None, None, :]
