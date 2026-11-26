import argparse
from pathlib import Path
from tgcn.data import load_bundle,make_loaders
from train import build_model
from tgcn.evaluation import evaluate,plot_results
from tgcn.utility import ROOT,device_for,load_checkpoint,save_json,seed_everything


def main():
    p=argparse.ArgumentParser(description='Evaluate a saved PyTorch checkpoint')
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--data-dir',default=None)
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    p.add_argument('--output-dir',default=None)
    p.add_argument('--plot-sensor',type=int,default=0)
    args=p.parse_args(); device=device_for(args.device);seed_everything(42)
    ckpt=load_checkpoint(args.checkpoint,device); cfg=ckpt['config']
    data_dir=args.data_dir or str(ROOT/'data'/cfg['dataset'])
    bundle=load_bundle(data_dir,cfg['dataset'],cfg['seq_len'],cfg['horizon'],
                       cfg['train_ratio'],cfg['val_ratio'],cfg['interval_min'],cfg['start_time'],
                       cfg.get('normalization','zscore'))
    loaders=make_loaders(bundle,cfg['seq_len'],cfg['horizon'],cfg['batch_size'])
    model=build_model(argparse.Namespace(**cfg),bundle).to(device)
    model.load_state_dict(ckpt['state_dict']);model.eval()
    out=Path(args.output_dir) if args.output_dir else Path(args.checkpoint).parent
    result,true,pred,indices=evaluate(model,loaders['test'],bundle,device,out,cfg['mape_min_speed'])
    result.update({'dataset':cfg['dataset'],'model':cfg['model'],'seed':cfg['seed'],
                   'best_epoch':ckpt['best_epoch'],'data_is_synthetic':cfg['dataset']=='demo',
                   'caution': ('SMOKE TEST ONLY, partial training batches; not for scientific comparison'
                               if cfg.get('max_train_batches') or cfg.get('max_val_batches') else '')})
    save_json(result,out/'metrics.json')
    plot_results(true,pred,indices,bundle.interval_min,out,args.plot_sensor)
    print('Metrics saved:',out/'metrics.json','\n',result['overall'])

if __name__=='__main__': main()
