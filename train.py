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
    p.add_argument('--dataset',choices=['losloop','sztaxi','demo','metrla','pemsbay'],default='losloop')
    p.add_argument('--data-dir',type=str,default=None)
    p.add_argument('--model',choices=['gru','gcn','tgcn','tgcn_plus','stid','gwnet','dcrnn','staeformer'],
                   default='tgcn')
    p.add_argument('--size',choices=['compact','full'],default='compact',
                   help='modern models: compact (CPU) or the papers\' full sizes (GPU).')
    p.add_argument('--seq-len',type=int,default=12)
    p.add_argument('--horizon',type=int,default=3)
    p.add_argument('--hidden-dim',type=int,default=64,
                   help='Paper: 64 for Los-loop, 100 for SZ-taxi.')
    p.add_argument('--graph-embed-dim',type=int,default=8)
    p.add_argument('--dropout',type=float,default=0.1)
    p.add_argument('--no-time-features',action='store_true')
    p.add_argument('--no-adaptive-graph',action='store_true')
    p.add_argument('--no-residual',action='store_true')
    p.add_argument('--time-features',action='store_true',
                   help='gru/tgcn: append sin/cos time-of-day & day-of-week to inputs.')
    p.add_argument('--residual',action='store_true',
                   help='gru/tgcn: predict change w.r.t. last observed speed.')
    p.add_argument('--adaptive-graph',action='store_true',help='tgcn: learn extra graph.')
    p.add_argument('--profile',action='store_true',
                   help='gru/tgcn: add the per-sensor time-of-day profile (train-only) as input '
                        'and its expected change to every forecast step.')
    p.add_argument('--self-sep',action='store_true',
                   help='tgcn: separate weights for the node itself and its neighbours ([x, Ax]W).')
    p.add_argument('--no-graph',action='store_true',
                   help='tgcn/tgcn_plus: replace adjacency by identity (ablation).')
    p.add_argument('--normalization',choices=['zscore','max'],default='zscore')
    p.add_argument('--protocol',choices=['strict','paper'],default='strict',
                   help="paper = authors' protocol (80/20, max-norm, epoch picked on TEST). "
                        'Only for reproducing Table 1; never for honest evaluation.')
    p.add_argument('--raw-loss',action='store_true',
                   help='Weight the loss by per-sensor variance = MSE in original units.')
    p.add_argument('--mask-zero',action='store_true',
                   help='Ignore zero targets (missing in SZ-taxi) in loss and selection.')
    p.add_argument('--track-test',action='store_true',
                   help='Log test RMSE each epoch (analysis of test-selection bias only).')
    p.add_argument('--train-ratio',type=float,default=0.7)
    p.add_argument('--val-ratio',type=float,default=0.1)
    p.add_argument('--interval-min',type=int,default=None)
    p.add_argument('--start-time',type=str,default=None)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--epochs',type=int,default=120)
    p.add_argument('--patience',type=int,default=20)
    p.add_argument('--lr',type=float,default=0.002)
    p.add_argument('--lr-schedule',choices=['none','cosine'],default='cosine',
                   help='cosine: lr decays from --lr to lr/20 over --epochs.')
    p.add_argument('--weight-decay',type=float,default=0.0001)
    p.add_argument('--loss',choices=['mse','huber','mae'],default='mse',
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


def build_model(args,bundle):
    a=vars(args)
    return make_model(args.model,bundle,args.hidden_dim,args.horizon,
                      a.get('no_time_features',False),a.get('no_adaptive_graph',False),
                      a.get('no_residual',False),args.dropout,args.graph_embed_dim,
                      seq_len=args.seq_len,time_features=a.get('time_features',False),
                      residual=a.get('residual',False),
                      adaptive_graph=a.get('adaptive_graph',False),
                      no_graph=a.get('no_graph',False),self_sep=a.get('self_sep',False),
                      size=a.get('size','compact'),profile=a.get('profile',False))


def run(args):
    if args.epochs<=0 or args.horizon<=0 or args.seq_len<=0:
        raise ValueError('epochs,horizon,seq_len must be positive')
    seed_everything(args.seed)
    device=device_for(args.device)
    args.data_dir = str(Path(args.data_dir) if args.data_dir else (ROOT/'data'/args.dataset))
    out=Path(args.result_dir) if args.result_dir else ROOT/'results'/args.dataset/args.model/f'seed_{args.seed}'
    out.mkdir(parents=True,exist_ok=True)
    bundle=load_bundle(args.data_dir,args.dataset,args.seq_len,args.horizon,
                       args.train_ratio,args.val_ratio,args.interval_min,args.start_time,
                       args.normalization,args.protocol)
    loaders=make_loaders(bundle,args.seq_len,args.horizon,args.batch_size,args.num_workers)
    model=build_model(args,bundle)
    print(f'Device={device}; data={args.dataset} {bundle.values.shape}; samples='+
          ', '.join(f'{s}:{len(loader.dataset)}' for s,loader in loaders.items()))
    save_json({'dataset':args.dataset,'data_shape':list(bundle.values.shape),
               'train_points':bundle.splits['train'][1],
               'val_points':bundle.splits['val'][1]-bundle.splits['val'][0],
               'test_points':bundle.splits['test'][1]-bundle.splits['test'][0],
               'training_window_samples':len(loaders['train'].dataset),
               'validation_window_samples':len(loaders['val'].dataset),
               'test_window_samples':len(loaders['test'].dataset),
               'normalization':f'train-only {args.normalization}',
               'start_time':args.start_time,'interval_min':bundle.interval_min,
               'args':vars(args)},out/'config.json')
    summary=train_model(model,loaders,device,args,out,bundle)
    checkpoint=load_checkpoint(out/'best.pt',device)
    model.load_state_dict(checkpoint['state_dict'])
    score,true,pred,indices=evaluate(model,loaders['test'],bundle,device,out,
                                      args.mape_min_speed)
    score.update({'dataset':args.dataset,'model':args.model,'seed':args.seed,
                  'best_epoch':checkpoint['best_epoch'],
                  'training':summary,
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
