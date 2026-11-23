"""Download public traffic datasets from the T-GCN author's GitHub repo.

Requires internet. Data files are not packaged as real data in the zip.
"""
import argparse
import time
import urllib.request
from pathlib import Path
from tgcn.utility import ROOT
from tgcn.data import read_csv_dataset

BASE='https://raw.githubusercontent.com/lehaifeng/T-GCN/master/data/'
FILES={
    'losloop':{'speed.csv':'los_speed.csv','adjacency.csv':'los_adj.csv'},
    'sztaxi':{'speed.csv':'sz_speed.csv','adjacency.csv':'sz_adj.csv'},
}

def main():
    p=argparse.ArgumentParser(description='Download official LOS-loop or SZ-taxi CSV')
    p.add_argument('--dataset',choices=FILES.keys(),default='losloop')
    p.add_argument('--out-dir',default=None)
    p.add_argument('--overwrite',action='store_true')
    args=p.parse_args(); out=Path(args.out_dir) if args.out_dir else ROOT/'data'/args.dataset
    out.mkdir(parents=True,exist_ok=True)
    for target, filename in FILES[args.dataset].items():
        dst=out/target
        if dst.exists() and not args.overwrite:
            print('Using existing:',dst);continue
        url=BASE+filename
        print('Downloading:',url,flush=True)
        last=None
        for retry in range(3):
            try:
                req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0 TGCN-Research'})
                with urllib.request.urlopen(req, timeout=90) as r:
                    content=r.read()
                if len(content)<500: raise ValueError('File too small / not valid CSV')
                (dst.parent/(dst.name+'.tmp')).write_bytes(content)
                (dst.parent/(dst.name+'.tmp')).replace(dst)
                last=None;break
            except Exception as exc:
                last=exc;time.sleep(1+retry)
        if last is not None:
            raise RuntimeError('Could not download '+url+'. Download manually from the URL, '
                               f'and place it as {dst}. Underlying error: {last}')
    values,adj,ids=read_csv_dataset(out)
    print(f'OK {args.dataset}: speed={values.shape}; graph={adj.shape}. Directory: {out}')

if __name__=='__main__':main()
