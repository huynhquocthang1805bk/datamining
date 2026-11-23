"""Create an explicitly SYNTHETIC offline demo dataset, never a research result."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tgcn.utility import ROOT


def generate(out_dir, nodes=10, steps=520, seed=123):
    if nodes<3 or steps<120: raise ValueError('Need >=3 nodes and >=120 steps')
    rng=np.random.default_rng(seed)
    # Symmetric ring network with two local connections per node.
    adj=np.zeros((nodes,nodes),dtype=np.float32)
    for i in range(nodes):
        adj[i,(i+1)%nodes]=1.; adj[(i+1)%nodes,i]=1.
    t=np.arange(steps)
    # Synthetic 15-min weekday speed oscillations with neighboring propagation.
    raw=np.zeros((steps,nodes),dtype=np.float32)
    for n in range(nodes):
        phase=(n/nodes)*2*np.pi
        morning=13*np.exp(-((t%96-32-n*.12)/8)**2)
        evening=11*np.exp(-((t%96-70-n*.10)/9)**2)
        raw[:,n]=(54+7*np.sin(2*np.pi*t/96+phase)
                  +3*np.sin(2*np.pi*t/(96*7)+phase/2)
                  -morning-evening+rng.normal(0,1.4,steps))
    for j in range(1,steps):
        raw[j]=.88*raw[j]+.12*(np.roll(raw[j-1],1)+np.roll(raw[j-1],-1))/2
    path=Path(out_dir);path.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(raw,columns=[f'S{n:03}' for n in range(nodes)]).to_csv(path/'speed.csv',index=False)
    pd.DataFrame(adj).to_csv(path/'adjacency.csv',header=False,index=False)
    (path/'README.txt').write_text('SYNTHETIC dataset for offline smoke tests only. NOT Los-loop or SZ-taxi.\n',encoding='utf-8')
    print(f'Wrote synthetic dataset {raw.shape} to {path}')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out-dir',default=str(ROOT/'data'/'demo'))
    p.add_argument('--nodes',type=int,default=10);p.add_argument('--steps',type=int,default=520)
    a=p.parse_args();generate(a.out_dir,a.nodes,a.steps)
