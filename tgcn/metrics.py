"""Forecast error metrics.

Besides MAE/RMSE/MAPE/WAPE the module implements the three extra metrics of
Zhao et al. (2019), computed exactly as in the authors' TensorFlow code:

    Accuracy = 1 - ||Y - Yhat||_F / ||Y||_F
    R2       = 1 - sum((Y - Yhat)^2) / sum((Y - mean(Y))^2)
    Var      = 1 - Var(Y - Yhat) / Var(Y)

``mask_zero=True`` ignores targets equal to 0. On SZ-taxi a 0 means "no taxi
observed in that 15-minute slot" (missing), not a real speed of 0 km/h.
"""
import numpy as np


def metrics(true, pred, min_mape_denom=5.0, mask_zero=False):
    true = np.asarray(true, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    if true.shape != pred.shape or not np.isfinite(pred).all():
        raise ValueError('Predictions and truth must have equal shape and be finite.')
    if mask_zero:
        keep = true != 0
        true, pred = true[keep], pred[keep]
    if true.size == 0:
        raise ValueError('No target left after masking.')
    e = pred - true
    mse = np.mean(e * e)
    mae = np.mean(np.abs(e))
    valid = np.abs(true) >= min_mape_denom
    mape = float(np.mean(np.abs(e[valid]) / np.abs(true[valid])) * 100) if valid.any() else None
    wape = float(np.sum(np.abs(e)) / max(float(np.sum(np.abs(true))), 1e-6) * 100)
    denom_ss = float(np.sum((true - true.mean()) ** 2))
    var_true = float(np.var(true))
    return {'MAE': float(mae), 'RMSE': float(np.sqrt(mse)),
            'MAPE_percent': mape, 'WAPE_percent': wape,
            'Accuracy': float(1 - np.linalg.norm(e) / max(np.linalg.norm(true), 1e-12)),
            'R2': float(1 - np.sum(e * e) / max(denom_ss, 1e-12)),
            'Var': float(1 - np.var(e) / max(var_true, 1e-12)),
            'mape_coverage': float(valid.mean()), 'n_predictions': int(true.size)}


def horizon_metrics(true, pred, interval_min, min_mape_denom=5.0, mask_zero=False):
    """true/pred: [samples, horizon, nodes].

    ``per_horizon``    : error of step k only (strict, used in this project).
    ``cumulative``     : error averaged over steps 1..k, which is how the
                         authors' code reports "15/30/45/60 min" (it flattens all
                         pre_len outputs before computing metrics).
    """
    result = {'overall': metrics(true, pred, min_mape_denom, mask_zero),
              'per_horizon': {}, 'cumulative': {}}
    for i in range(true.shape[1]):
        key = f'{(i + 1) * interval_min}min'
        result['per_horizon'][key] = metrics(true[:, i, :], pred[:, i, :], min_mape_denom, mask_zero)
        result['cumulative'][key] = metrics(true[:, :i + 1, :], pred[:, :i + 1, :], min_mape_denom, mask_zero)
    return result
