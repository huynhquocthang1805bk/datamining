from pathlib import Path
import numpy as np
import torch
from .data import inverse_transform
from .metrics import horizon_metrics
from .utility import save_json


@torch.no_grad()
def predict_loader(model, loader, device):
    model.eval()
    p, y, times = [], [], []
    for x, tf, target, t in loader:
        x=x.to(device); tf=tf.to(device)
        yp=model(x,tf)
        p.append(yp.cpu().numpy())
        y.append(target.numpy()); times.append(t.numpy())
    return np.concatenate(y), np.concatenate(p), np.concatenate(times)


def evaluate(model, loader, bundle, device, output_dir, min_mape_denom=5.,
             save_predictions=True):
    true_scaled,pred_scaled,t = predict_loader(model, loader, device)
    true=inverse_transform(true_scaled,bundle)
    pred=inverse_transform(pred_scaled,bundle)
    scores=horizon_metrics(true,pred,bundle.interval_min,min_mape_denom)
    scores['n_samples'] = int(len(t))
    scores['n_nodes'] = int(len(bundle.sensor_ids))
    output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True)
    save_json(scores,output_dir/'metrics.json')
    if save_predictions:
        np.savez_compressed(output_dir/'test_predictions.npz',
                            y_true=true.astype(np.float32), y_pred=pred.astype(np.float32),
                            time_indices=t, sensor_ids=np.array(bundle.sensor_ids))
    return scores,true,pred,t


def plot_results(true,pred,steps,interval_min,output_dir,sensor_idx=0):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True)
    if sensor_idx >= true.shape[2]:
        raise ValueError('plot sensor index out of range')
    # Plot for fixed horizon step 1. Each marker is one consecutive timestamp,
    # thus the line represents genuine time-series, not arbitrary samples.
    n=min(250,len(steps))
    x=steps[:n]*interval_min/60
    fig,ax=plt.subplots(figsize=(11,4))
    ax.plot(x,true[:n,0,sensor_idx],label='Actual',linewidth=1.6)
    ax.plot(x,pred[:n,0,sensor_idx],label='Forecast',linewidth=1.2,alpha=.86)
    ax.set(xlabel='Elapsed hours since dataset origin', ylabel='Speed (dataset unit)',
           title=f'Sensor {sensor_idx}: {interval_min}-minute-ahead forecast')
    ax.grid(alpha=.2); ax.legend(); fig.tight_layout()
    fig.savefig(output_dir/'forecast_sensor.png',dpi=175); plt.close(fig)
    fig,ax=plt.subplots(figsize=(5.6,5))
    ax.scatter(true[:,0,:].ravel()[::5],pred[:,0,:].ravel()[::5],s=5,alpha=.25)
    bound=max(float(true.max()),float(pred.max()))
    ax.plot([0,bound],[0,bound],'k--',linewidth=1)
    ax.set(xlabel='Actual',ylabel='Prediction',title='Predicted vs. actual (horizon 1)')
    fig.tight_layout(); fig.savefig(output_dir/'scatter.png',dpi=175); plt.close(fig)
