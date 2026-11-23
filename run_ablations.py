"""Run controlled ablations on T-GCN+: same data and seed, one feature off."""
import argparse
import subprocess
import sys
from tgcn.utility import ROOT


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',choices=['losloop','sztaxi','demo'],default='losloop')
    p.add_argument('--epochs',type=int,default=50)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--horizon',type=int,default=3)
    p.add_argument('--seq-len',type=int,default=12)
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    a=p.parse_args()
    conditions=[('all_features',[]),('no_time',['--no-time-features']),
                ('no_adaptive_graph',['--no-adaptive-graph']),
                ('no_residual',['--no-residual']),('huber_loss',['--loss','huber'])]
    for label,flags in conditions:
        cmd=[sys.executable,str(ROOT/'train.py'),'--dataset',a.dataset,
             '--model','tgcn_plus','--epochs',str(a.epochs),
             '--seed',str(a.seed),'--seq-len',str(a.seq_len),
             '--horizon',str(a.horizon),'--device',a.device,
             '--result-dir',str(ROOT/'results'/a.dataset/'ablations'/label),*flags]
        print('EXEC:',label,flush=True)
        subprocess.run(cmd,cwd=ROOT,check=True)
    subprocess.run([sys.executable,str(ROOT/'compare.py'),'--dataset',a.dataset],
                   cwd=ROOT,check=True)
    subprocess.run([sys.executable,str(ROOT/'make_report.py'),'--dataset',a.dataset],
                   cwd=ROOT,check=True)

if __name__=='__main__':main()
