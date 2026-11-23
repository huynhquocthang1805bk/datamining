import json
import os
import random
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(min(os.cpu_count() or 1, 4))


def device_for(request):
    if request == 'auto': return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if request == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA không khả dụng. Thử --device cpu')
    return torch.device(request)


def save_json(obj, path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')


def load_checkpoint(path, device='cpu'):
    return torch.load(path, map_location=device, weights_only=True)
