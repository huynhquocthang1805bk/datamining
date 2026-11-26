from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from .data import target_mask
from .utility import save_json


def _loss_fn(name):
    if name == 'huber':
        return lambda p, y: nn.functional.smooth_l1_loss(p, y, beta=1.0, reduction='none')
    if name == 'mae':
        return lambda p, y: (p - y).abs()
    return lambda p, y: (p - y) ** 2


def masked_mean(err, mask):
    if mask is None:
        return err.mean()
    mask = mask.to(err.dtype)
    return (err * mask).sum() / mask.sum().clamp_min(1.0)


@torch.no_grad()
def _raw_rmse(model, loader, bundle, device, mask_zero):
    """RMSE in original speed units (for monitoring only)."""
    std = torch.as_tensor(bundle.stds, device=device)
    se, cnt = 0.0, 0.0
    for x, tf, y, t in loader:
        p = model(x.to(device), tf.to(device))
        e = ((p - y.to(device)) * std) ** 2
        if mask_zero:
            m = target_mask(bundle, t, y.shape[1]).to(device)
            se += float((e * m).sum()); cnt += float(m.sum())
        else:
            se += float(e.sum()); cnt += e.numel()
    return (se / max(cnt, 1)) ** 0.5


def train_model(model, loaders, device, args, output_dir, bundle=None):
    """Adam(W) + early stopping on validation MSE. The test loader is never used
    for selection; with --track-test its RMSE is only *logged* per epoch so that
    the optimistic bias of the authors' protocol (pick the epoch with the lowest
    test RMSE) can be measured afterwards."""
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs,
                                                            eta_min=args.lr / 20)
                 if getattr(args, 'lr_schedule', 'none') == 'cosine' else None)
    criterion = _loss_fn(args.loss)
    mask_zero = bool(getattr(args, 'mask_zero', False))
    # raw_loss: weight each sensor by std^2 so that the z-space MSE equals the MSE in
    # original speed units (up to a constant) -> congested, high-variance sensors count more
    node_w = None
    if getattr(args, 'raw_loss', False) and bundle is not None:
        sd = torch.as_tensor(bundle.stds, dtype=torch.float32)
        node_w = (sd ** 2 / (sd ** 2).mean()).to(device)
    track_test = bool(getattr(args, 'track_test', False)) and bundle is not None
    history = []; best = float('inf'); stale = 0; best_epoch = -1
    model = model.to(device)
    # everything needed to use the checkpoint without re-reading the training data
    data_stats = None if bundle is None else {
        'means': torch.as_tensor(bundle.means), 'stds': torch.as_tensor(bundle.stds),
        'adj_normalized': torch.as_tensor(bundle.adj), 'sensor_ids': list(bundle.sensor_ids),
        'interval_min': int(bundle.interval_min), 'normalization': bundle.normalization,
        'start_time': str(bundle.timestamps[0]) if bundle.timestamps is not None else None}
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train(); losses = []
        for batch_idx, (x, tf, y, t) in enumerate(loaders['train']):
            if args.max_train_batches and batch_idx >= args.max_train_batches: break
            x = x.to(device); tf = tf.to(device); y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(x, tf)
            m = target_mask(bundle, t, y.shape[1]).to(device) if (mask_zero and bundle is not None) else None
            err = criterion(output, y)
            if node_w is not None:
                err = err * (node_w if args.loss == 'mse' else node_w.sqrt())
            loss = masked_mean(err, m)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite loss: check learning rate, data, normalization.')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step(); losses.append(loss.item())
        model.eval(); val_se = 0.; n = 0.
        with torch.no_grad():
            for batch_idx, (x, tf, y, t) in enumerate(loaders['val']):
                if args.max_val_batches and batch_idx >= args.max_val_batches: break
                p = model(x.to(device), tf.to(device))
                e = (p - y.to(device)) ** 2
                if node_w is not None:
                    e = e * node_w
                if mask_zero and bundle is not None:
                    m = target_mask(bundle, t, y.shape[1]).to(device)
                    val_se += float((e * m).sum()); n += float(m.sum())
                else:
                    val_se += float(e.sum()); n += e.numel()
        score = val_se / max(n, 1)
        train_loss = sum(losses) / len(losses)
        row = {'epoch': epoch, 'train_loss': train_loss, 'val_mse_scaled': score,
               'lr': optimizer.param_groups[0]['lr']}
        if scheduler is not None:
            scheduler.step()
        if track_test:
            row['test_rmse_raw_monitor'] = _raw_rmse(model, loaders['test'], bundle, device, mask_zero)
            row['val_rmse_raw'] = _raw_rmse(model, loaders['val'], bundle, device, mask_zero)
        history.append(row)
        if epoch == 1 or epoch % 10 == 0:
            print(f'[{args.model}] epoch {epoch:03d}: train_loss={train_loss:.5f} val_mse={score:.5f}'
                  + (f" test_rmse={row['test_rmse_raw_monitor']:.4f}" if track_test else ''), flush=True)
        if score < best - 1e-7:
            best = score; stale = 0; best_epoch = epoch
            state = {'state_dict': model.state_dict(), 'config': vars(args),
                     'best_val_mse_scaled': float(best), 'best_epoch': epoch,
                     'data_stats': data_stats}
            torch.save(state, out / 'best.pt')
        else:
            stale += 1
        if stale >= args.patience:
            print(f'Early stopping after {epoch} epochs (best epoch {best_epoch})', flush=True)
            break
    elapsed = time.time() - t0
    pd.DataFrame(history).to_csv(out / 'history.csv', index=False)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot([r['epoch'] for r in history], [r['val_mse_scaled'] for r in history], label='Validation MSE')
        ax.plot([r['epoch'] for r in history], [r['train_loss'] for r in history], label='Train loss', alpha=.7)
        ax.axvline(best_epoch, color='k', ls=':', lw=1)
        ax.set(xlabel='Epoch', ylabel='Normalized loss', title='Learning curve')
        ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
        fig.savefig(out / 'learning_curve.png', dpi=140); plt.close(fig)
    except ImportError:
        pass
    summary = {'best_epoch': best_epoch, 'best_val_mse_scaled': float(best),
               'epochs_run': len(history), 'train_seconds': elapsed,
               'n_parameters': int(sum(p.numel() for p in model.parameters()))}
    if track_test:
        h = pd.DataFrame(history)
        k = int(h.test_rmse_raw_monitor.idxmin())
        summary.update({'oracle_epoch_min_test_rmse': int(h.epoch[k]),
                        'oracle_test_rmse': float(h.test_rmse_raw_monitor[k]),
                        'valselected_test_rmse_monitor': float(h.test_rmse_raw_monitor[best_epoch - 1])})
    save_json(summary, out / 'training_summary.json')
    return summary
