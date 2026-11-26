"""Plot test errors for trained SZTaxi GPU runs."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--include-zero', action='store_true')
    args = parser.parse_args()
    rows = []
    for path in sorted((ROOT / 'results/sztaxi/modern_gpu').glob('*/seed_*/metrics.json')):
        saved = json.loads(path.read_text(encoding='utf-8'))
        scores = saved if args.include_zero else saved['masked_nonzero']
        horizons = {'overall': scores['overall'], **scores['per_horizon']}
        for horizon, metrics in horizons.items():
            rows.append({'model': path.parent.parent.name, 'seed': path.parent.name,
                         'horizon': horizon, **{key: metrics[key] for key in ('RMSE', 'MAE', 'R2')}})
    if not rows:
        parser.exit(1, 'No completed SZTaxi GPU runs found.\n')

    summary = pd.DataFrame(rows).groupby(['model', 'horizon']).agg(
        seeds=('seed', 'nunique'), RMSE=('RMSE', 'mean'), RMSE_sd=('RMSE', 'std'),
        MAE=('MAE', 'mean'), MAE_sd=('MAE', 'std'), R2=('R2', 'mean')).reset_index()
    summary['minutes'] = summary['horizon'].map(
        lambda horizon: 0 if horizon == 'overall' else int(horizon.removesuffix('min')))
    overall = summary[summary.minutes.eq(0)].sort_values('RMSE')
    print(overall.drop(columns=['horizon', 'minutes']).round(4).to_string(index=False))

    scope = 'all' if args.include_zero else 'nonzero'
    stem = f'sztaxi_gpu_test_{scope}'
    table = ROOT / 'reports/tables' / f'{stem}.csv'
    table.parent.mkdir(parents=True, exist_ok=True)
    summary.sort_values(['model', 'minutes']).drop(columns='minutes').to_csv(
        table, index=False, encoding='utf-8-sig', float_format='%.4f')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
    axes[0].errorbar(overall.RMSE, overall.model, xerr=overall.RMSE_sd.fillna(0),
                     fmt='o', capsize=4, color='#2166ac')
    axes[0].invert_yaxis()
    axes[0].set(title='Overall RMSE (mean and seed SD)', xlabel='RMSE')
    axes[0].grid(axis='x', alpha=.25)
    steps = summary[summary.minutes.gt(0)].sort_values('minutes')
    for model, points in steps.groupby('model', sort=False):
        axes[1].plot(points.minutes, points.RMSE, marker='o', label=model)
    axes[1].set(title='RMSE by forecast horizon', xlabel='Minutes ahead',
                ylabel='RMSE', xticks=sorted(steps.minutes.unique()))
    axes[1].grid(alpha=.25)
    axes[1].legend(fontsize=8)
    fig.suptitle(f'SZTaxi GPU test | {scope} targets')
    figure = ROOT / 'reports/figures' / f'{stem}.png'
    figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure, dpi=180)
    plt.close(fig)
    print(table.relative_to(ROOT))
    print(figure.relative_to(ROOT))


if __name__ == '__main__':
    main()
