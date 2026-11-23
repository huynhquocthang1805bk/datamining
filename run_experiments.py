"""Fair model comparison: identical dataset, split, horizons and seeds."""
import argparse
import subprocess
import sys
from pathlib import Path
from tgcn.utility import ROOT


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',choices=['losloop','sztaxi','demo'],default='losloop')
    p.add_argument('--epochs',type=int,default=50)
    p.add_argument('--seeds',nargs='+',type=int,default=[42])
    p.add_argument('--horizon',type=int,default=3)
    p.add_argument('--seq-len',type=int,default=12)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--models',nargs='+',choices=['gru','tgcn','tgcn_plus'],
                   default=['gru','tgcn','tgcn_plus'])
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    p.add_argument('--max-train-batches',type=int,default=0)
    p.add_argument('--max-val-batches',type=int,default=0)
    args=p.parse_args()
    for seed in args.seeds:
        for model in args.models:
            command=[sys.executable,str(ROOT/'train.py'), '--dataset',args.dataset,
                     '--model',model,'--epochs',str(args.epochs),
                     '--seed',str(seed),'--horizon',str(args.horizon),
                     '--seq-len',str(args.seq_len), '--batch-size',str(args.batch_size),
                     '--device',args.device,
                     '--max-train-batches',str(args.max_train_batches),
                     '--max-val-batches',str(args.max_val_batches)]
            print('EXEC:', ' '.join(command),flush=True)
            subprocess.run(command,cwd=ROOT,check=True)
    subprocess.run([sys.executable,str(ROOT/'compare.py'), '--dataset',args.dataset],
                   cwd=ROOT, check=True)

if __name__=='__main__':main()
