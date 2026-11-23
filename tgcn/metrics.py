import numpy as np


def metrics(true, pred, min_mape_denom=5.0):
    true = np.asarray(true, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    if true.shape != pred.shape or not np.isfinite(pred).all():
        raise ValueError('Predictions and truth must have equal shape and be finite.')
    e = pred - true
    mse = np.mean(e*e)
    mae = np.mean(np.abs(e))
    valid = np.abs(true) >= min_mape_denom
    mape = float(np.mean(np.abs(e[valid]) / np.abs(true[valid]))*100) if valid.any() else None
    wape = float(np.sum(np.abs(e))/max(float(np.sum(np.abs(true))), 1e-6)*100)
    return {'MAE': float(mae), 'RMSE': float(np.sqrt(mse)),
            'MAPE_percent': mape, 'WAPE_percent': wape,
            'mape_coverage': float(valid.mean()), 'n_predictions': int(true.size)}


def horizon_metrics(true, pred, interval_min, min_mape_denom=5.0):
    result = {'overall': metrics(true,pred,min_mape_denom), 'per_horizon': {}}
    for i in range(true.shape[1]):
        result['per_horizon'][f'{(i+1)*interval_min}min'] = metrics(
            true[:,i,:],pred[:,i,:],min_mape_denom)
    return result
