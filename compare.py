"""Compare actual completed experiments only (never impute missing metrics)."""
import argparse
import json
from pathlib import Path
import pandas as pd
from tgcn.utility import ROOT


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',choices=['losloop','sztaxi','demo'],default='losloop')
    p.add_argument('--results-dir',default=str(ROOT/'results'))
    args=p.parse_args();base=Path(args.results_dir)/args.dataset
    rows=[]
    for f in sorted(base.glob('**/metrics.json')):
        m=json.loads(f.read_text(encoding='utf-8'))
        if m.get('dataset')!=args.dataset:continue
        partial=bool(m.get('caution'))
        config_path=f.parent/'config.json'
        if config_path.is_file():
            config=json.loads(config_path.read_text(encoding='utf-8'))
            partial = partial or bool(config.get('args',{}).get('max_train_batches')) or bool(config.get('args',{}).get('max_val_batches'))
        info={'model':m['model'],'seed':m.get('seed'), 'run':str(f.parent.relative_to(base)),
              'is_synthetic':m.get('data_is_synthetic',False),
              'is_partial':partial}
        info.update({k:v for k,v in m['overall'].items() if k in ('MAE','RMSE','MAPE_percent','WAPE_percent')})
        rows.append(info)
    if not rows:
        raise SystemExit('No metrics.json found for this dataset. Train models first.')
    df=pd.DataFrame(rows).sort_values(['model','seed'])
    base.mkdir(parents=True,exist_ok=True)
    df.to_csv(base/'comparison.csv',index=False)
    # Exclude partial-batch runs from scientific comparisons.
    clean=df.loc[~df.is_partial]
    if len(clean):
        agg=clean.groupby('model')[['MAE','RMSE','MAPE_percent','WAPE_percent']].agg(['mean','std','count'])
        agg.to_csv(base/'comparison_aggregated.csv')
    print(df.to_string(index=False))
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(8,4))
        target=clean if len(clean) else df
        group=target.groupby('model')['RMSE'].mean().sort_values()
        ax.bar(group.index,group.values)
        ax.set(ylabel='Test RMSE (speed unit)',title=f'Model comparison — {args.dataset}')
        fig.tight_layout();fig.savefig(base/'comparison.png',dpi=180);plt.close(fig)
    except ImportError:
        pass
    print('Saved',base/'comparison.csv')

if __name__=='__main__':main()
