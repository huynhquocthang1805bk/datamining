"""Train one model and evaluate its best validation checkpoint on held-out test data."""
import argparse
from pathlib import Path
from tgcn.data import load_bundle,make_loaders
from tgcn.models import make_model
from tgcn.training import train_model
from tgcn.evaluation import evaluate,plot_results
from tgcn.utility import ROOT,seed_everything,device_for,load_checkpoint,save_json


def build_parser():
    p=argparse.ArgumentParser(description='Traffic forecasting with GRU, T-GCN, T-GCN+')
    p.add_argument('--dataset',choices=['losloop','sztaxi','demo'],default='losloop')
    p.add_argument('--data-dir',type=str,default=None)
    p.add_argument('--model',choices=['gru','tgcn','tgcn_plus'],default='tgcn')
    p.add_argument('--seq-len',type=int,default=12)
    p.add_argument('--horizon',type=int,default=3)
    p.add_argument('--hidden-dim',type=int,default=32)
    p.add_argument('--graph-embed-dim',type=int,default=8)
    p.add_argument('--dropout',type=float,default=0.1)
    p.add_argument('--no-time-features',action='store_true')
    p.add_argument('--no-adaptive-graph',action='store_true')
    p.add_argument('--no-residual',action='store_true')
    p.add_argument('--train-ratio',type=float,default=0.7)
    p.add_argument('--val-ratio',type=float,default=0.1)
    p.add_argument('--interval-min',type=int,default=None)
    p.add_argument('--start-time',type=str,default=None)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--epochs',type=int,default=50)
    p.add_argument('--patience',type=int,default=10)
    p.add_argument('--lr',type=float,default=0.001)
    p.add_argument('--weight-decay',type=float,default=0.0001)
    p.add_argument('--loss',choices=['mse','huber'],default='mse',
                   help='Use MSE for fair main comparison. Test Huber separately.')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--num-workers',type=int,default=0)
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    p.add_argument('--result-dir',type=str,default=None)
    p.add_argument('--plot-sensor',type=int,default=0)
    p.add_argument('--max-train-batches',type=int,default=0,
                   help='For smoke tests only. 0 = full data.')
    p.add_argument('--max-val-batches',type=int,default=0,
                   help='For smoke tests only. 0 = full data.')
    p.add_argument('--mape-min-speed',type=float,default=5.,
                   help='Exclude |truth| below this threshold in MAPE.')
    return p


def run(args):
    if args.epochs<=0 or args.horizon<=0 or args.seq_len<=0:
        raise ValueError('epochs,horizon,seq_len must be positive')
    seed_everything(args.seed)
    device=device_for(args.device)
    args.data_dir = str(Path(args.data_dir) if args.data_dir else (ROOT/'data'/args.dataset))
    out=Path(args.result_dir) if args.result_dir else ROOT/'results'/args.dataset/args.model/f'seed_{args.seed}'
    out.mkdir(parents=True,exist_ok=True)
    bundle=load_bundle(args.data_dir,args.dataset,args.seq_len,args.horizon,
                       args.train_ratio,args.val_ratio,args.interval_min,args.start_time)
    loaders=make_loaders(bundle,args.seq_len,args.horizon,args.batch_size,args.num_workers)
    model=make_model(args.model,bundle,args.hidden_dim,args.horizon,
                     args.no_time_features,args.no_adaptive_graph,args.no_residual,
                     args.dropout,args.graph_embed_dim)
    print(f'Device={device}; data={args.dataset} {bundle.values.shape}; samples='+
          ', '.join(f'{s}:{len(loader.dataset)}' for s,loader in loaders.items()))
    save_json({'dataset':args.dataset,'data_shape':list(bundle.values.shape),
               'train_points':bundle.splits['train'][1],
               'val_points':bundle.splits['val'][1]-bundle.splits['val'][0],
               'test_points':bundle.splits['test'][1]-bundle.splits['test'][0],
               'training_window_samples':len(loaders['train'].dataset),
               'validation_window_samples':len(loaders['val'].dataset),
               'test_window_samples':len(loaders['test'].dataset),
               'normalization':'train-only per-sensor zscore',
               'start_time':args.start_time,'interval_min':bundle.interval_min,
               'args':vars(args)},out/'config.json')
    train_model(model,loaders,device,args,out)
    checkpoint=load_checkpoint(out/'best.pt',device)
    model.load_state_dict(checkpoint['state_dict'])
    score,true,pred,indices=evaluate(model,loaders['test'],bundle,device,out,
                                      args.mape_min_speed)
    score.update({'dataset':args.dataset,'model':args.model,'seed':args.seed,
                  'best_epoch':checkpoint['best_epoch'],
                  'data_is_synthetic':args.dataset=='demo',
                  'caution':('SMOKE TEST ONLY, partial training batches; not for scientific comparison'
                            if args.max_train_batches or args.max_val_batches else '')})
    save_json(score,out/'metrics.json')
    plot_results(true,pred,indices,bundle.interval_min,out,args.plot_sensor)
    print('Saved:',out,'\nTest MAE:',round(score['overall']['MAE'],4),
          'RMSE:',round(score['overall']['RMSE'],4))
    return score

if __name__=='__main__':
    run(build_parser().parse_args())
