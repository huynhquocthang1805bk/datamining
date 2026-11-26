"""Phase 3 — effect of input window, hidden units and of each added component.

All comparisons in this script use the same seed (42) so that a difference is
caused by the configuration, not by the random initialisation.

    python analysis/tuning_ablation.py --dataset losloop
"""
import argparse
import numpy as np
import pandas as pd
from common import FIG, TAB, run_dirs, df_to_markdown
from tgcn import plotstyle as ps
import matplotlib.pyplot as plt


def row(r, label, ks):
    m = r['metrics']
    out = {'config': label, 'RMSE_all': m['overall']['RMSE'], 'MAE_all': m['overall']['MAE'],
           'best_epoch': m.get('best_epoch'), 'n_params': r['summary'].get('n_parameters'),
           'train_s': r['summary'].get('train_seconds')}
    for k in ks:
        out[f'RMSE_{k}'] = m['per_horizon'][k]['RMSE']; out[f'MAE_{k}'] = m['per_horizon'][k]['MAE']
    return out


def main():
    p = argparse.ArgumentParser(); p.add_argument('--dataset', default='losloop')
    ds = p.parse_args().dataset
    runs = run_dirs(ds)
    ks = ['15min', '30min', '60min']
    s42 = {(r['group'], r['name']): r for r in runs if r['seed'] == 42}
    base = s42.get(('main', 'tgcn'))
    if base is None:
        raise SystemExit('need main/tgcn seed 42')
    L0, H0 = 12, base['metrics'].get('training', {}).get('n_parameters')
    # ---------- tuning ----------
    tun = [row(base, 'T-GCN L=12, h=64 (mặc định)', ks)]
    for (g, n), r in sorted(s42.items()):
        if g == 'tuning':
            tun.append(row(r, n, ks))
    if ('improve', 'tgcn_res_time') in s42:
        tun.append(row(s42[('improve', 'tgcn_res_time')], 'tgcn_res_time L=12, h=64', ks))
    tdf = pd.DataFrame(tun); tdf.to_csv(TAB / f'tuning_{ds}.csv', index=False)
    print(tdf.round(3).to_string())
    # ---------- ablation (one component at a time) ----------
    labels = {('main', 'gru'): 'GRU (không đồ thị)', ('main', 'tgcn'): 'T-GCN (gốc)',
              ('variants', 'tgcn_selfloop1'): 'T-GCN, self-loop = 1 (bản cài đặt đầu)',
              ('variants', 'tgcn_time'): '+ thời gian', ('variants', 'tgcn_res'): '+ residual',
              ('variants', 'tgcn_selfsep'): '+ tách self/láng giềng',
              ('improve', 'tgcn_res_time'): '+ residual + thời gian (T-GCN-RT)',
              ('variants', 'tgcn_res_time_nograph'): 'T-GCN-RT bỏ đồ thị (Â = I)',
              ('improve', 'gru_res_time'): 'GRU + residual + thời gian',
              ('variants', 'tgcn_selfsep_res_time'): 'T-GCN-RT + tách self/láng giềng',
              ('variants', 'tgcn_plus'): 'T-GCN-RT + đồ thị thích nghi (T-GCN+)',
              ('variants', 'tgcn_res_time_huber'): 'T-GCN-RT + Huber loss',
              ('variants', 'tgcn_res_time_rawloss'): 'T-GCN-RT + loss đơn vị gốc',
              ('variants', 'gru_res_time_rawloss'): 'GRU-RT + loss đơn vị gốc',
              ('baselines', 'naive'): 'Naive (giữ giá trị cuối)'}
    ab = [row(s42[k], v, ks) for k, v in labels.items() if k in s42]
    adf = pd.DataFrame(ab)
    b0 = adf.set_index('config').loc['T-GCN (gốc)']
    for c in ('RMSE_all', 'MAE_all'):
        adf[f'Δ{c}_vs_TGCN_%'] = 100 * (adf[c] - b0[c]) / b0[c]
    adf.to_csv(TAB / f'ablation_{ds}.csv', index=False)
    (TAB / f'ablation_{ds}.md').write_text(df_to_markdown(adf[['config', 'RMSE_all', 'MAE_all', 'RMSE_15min', 'MAE_15min',
                                                               'RMSE_60min', 'MAE_60min', 'ΔRMSE_all_vs_TGCN_%',
                                                               'ΔMAE_all_vs_TGCN_%']].round(3), index=False),
                                           encoding='utf-8')
    print(adf.round(3).to_string())
    # figure: ablation bars
    fig, ax = plt.subplots(figsize=(10, 0.42 * len(adf) + 1.4))
    y = np.arange(len(adf))[::-1]
    ax.barh(y + .2, adf['ΔRMSE_all_vs_TGCN_%'], .38, color=ps.SERIES[0], label='RMSE')
    ax.barh(y - .2, adf['ΔMAE_all_vs_TGCN_%'], .38, color=ps.SERIES[1], label='MAE')
    ax.axvline(0, color=ps.INK2, lw=.9)
    ax.set(yticks=y, yticklabels=adf.config, xlabel='Thay đổi so với T-GCN gốc (%) — âm là tốt hơn',
           title=f'{ds}: đóng góp của từng thành phần (seed 42, trung bình 12 tầm dự báo)')
    ax.legend(loc='lower right'); ax.grid(axis='y', visible=False)
    fig.tight_layout(); fig.savefig(FIG / f'ablation_{ds}.png'); plt.close(fig)
    # figure: tuning (2x2 small multiples: rows = metric, cols = knob)
    fig, axes = plt.subplots(2, 2, figsize=(11, 5.6), sharex='col')
    seq = [(6, s42.get(('tuning', 'tgcn_seq6'))), (12, base), (24, s42.get(('tuning', 'tgcn_seq24')))]
    seq_rt = [(6, s42.get(('tuning', 'tgcn_res_time_seq6'))), (12, s42.get(('improve', 'tgcn_res_time'))),
              (24, s42.get(('tuning', 'tgcn_res_time_seq24')))]
    hid = [(32, s42.get(('tuning', 'tgcn_h32'))), (64, base), (100, s42.get(('tuning', 'tgcn_h100')))]
    for col, series, xl, xt in ((0, [('tgcn', seq), ('tgcn_res_time', seq_rt)], 'Cửa sổ đầu vào L (bước 5 phút)', [6, 12, 24]),
                                (1, [('tgcn', hid)], 'Số hidden units', [32, 64, 100])):
        for ri, key in ((0, 'RMSE'), (1, 'MAE')):
            ax = axes[ri][col]
            for name, pts in series:
                pts = [(x, r) for x, r in pts if r is not None]
                if not pts: continue
                ax.plot([x for x, _ in pts], [r['metrics']['overall'][key] for _, r in pts], marker='o', ms=6,
                        color=ps.color(name), label=ps.label(name))
            ax.set(xticks=xt, ylabel=f'{key} test (TB 12 tầm)')
            if ri == 1: ax.set_xlabel(xl)
    axes[0][0].set_title('Độ dài cửa sổ đầu vào'); axes[0][1].set_title('Số hidden units (T-GCN gốc)')
    axes[0][0].legend(fontsize=8)
    fig.suptitle(f'{ds}: tinh chỉnh siêu tham số (seed 42) — mọi khác biệt ≤ ~1.3%', fontsize=11.5)
    fig.tight_layout(); fig.savefig(FIG / f'tuning_{ds}.png'); plt.close(fig)


if __name__ == '__main__':
    main()
