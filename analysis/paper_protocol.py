"""Replication of the authors' protocol (80/20, max-norm, epoch picked on TEST).

Plots test RMSE per epoch for GRU and T-GCN next to the values reported in
Table 1 and the persistence baseline computed on exactly the same windows.

    python analysis/paper_protocol.py
"""
import json
import numpy as np
import pandas as pd
from common import ROOT, FIG, TAB, PAPER, df_to_markdown
from tgcn import plotstyle as ps
import matplotlib.pyplot as plt

SETUP = {'losloop': (3, 15), 'sztaxi': (1, 15)}   # pre_len, minutes reported


def naive_same_windows(ds, pre_len, seq_len=12):
    v = pd.read_csv(ROOT / 'data' / ds / 'speed.csv').to_numpy(float)
    te = int(len(v) * 0.8); test = v[te:]
    X, Y = [], []
    for i in range(len(test) - seq_len - pre_len):
        X.append(test[i:i + seq_len]); Y.append(test[i + seq_len:i + seq_len + pre_len])
    X, Y = np.array(X), np.array(Y)
    e = np.repeat(X[:, -1:], pre_len, 1) - Y
    w = np.repeat(X.mean(1, keepdims=True), pre_len, 1) - Y
    return (float(np.sqrt((e ** 2).mean())), float(np.abs(e).mean()),
            float(np.sqrt((w ** 2).mean())), float(np.abs(w).mean()))


def main():
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    for ax, ds in zip(axes, ('losloop', 'sztaxi')):
        pre, mins = SETUP[ds]
        nr, na, hr, ha = naive_same_windows(ds, pre)
        rows.append({'dataset': ds, 'model': 'naive', 'epochs': 0, 'RMSE': nr, 'MAE': na})
        rows.append({'dataset': ds, 'model': 'ha_window', 'epochs': 0, 'RMSE': hr, 'MAE': ha,
                     'paper_RMSE': PAPER[ds]['ha_window'][mins][0], 'paper_MAE': PAPER[ds]['ha_window'][mins][1]})
        for i, m in enumerate(('gru', 'tgcn')):
            d = ROOT / 'results' / ds / 'paper_protocol' / m / 'seed_42'
            if not (d / 'metrics.json').is_file():
                continue
            met = json.loads((d / 'metrics.json').read_text())
            h = pd.read_csv(d / 'history.csv')
            o = met['overall']
            rows.append({'dataset': ds, 'model': m, 'epochs': len(h), 'best_epoch': met['best_epoch'],
                         'RMSE': o['RMSE'], 'MAE': o['MAE'], 'Accuracy': o['Accuracy'], 'R2': o['R2'],
                         'paper_RMSE': PAPER[ds][m][mins][0], 'paper_MAE': PAPER[ds][m][mins][1],
                         'paper_Accuracy': PAPER[ds][m][mins][2], 'paper_R2': PAPER[ds][m][mins][3],
                         'RMSE_epoch_half': float(h.test_rmse_raw_monitor.iloc[len(h) // 2 - 1])})
            ax.plot(h.epoch, h.test_rmse_raw_monitor, color=ps.color(m), lw=1.6, label=f'{ps.label(m)} (của chúng tôi)')
            ax.axhline(PAPER[ds][m][mins][0], color=ps.color(m), ls=':', lw=1.4)
            ax.text(len(h) * 0.12, PAPER[ds][m][mins][0], f'bài báo {ps.label(m)}: {PAPER[ds][m][mins][0]:.2f}',
                    ha='left', va='bottom' if m == 'gru' else 'top', fontsize=7.5, color=ps.color(m))
        ax.axhline(nr, color=ps.NEUTRAL, ls='--', lw=1.2)
        ax.text(len(h) * 0.55, nr, f'persistence: {nr:.2f}', fontsize=7.5, color=ps.INK2, va='bottom')
        lo = min(r['RMSE'] for r in rows if r['dataset'] == ds) * 0.93
        ax.set_ylim(lo, lo * 1.45)
        ax.set(title=f'{ds}: "{mins} phút", giao thức của tác giả', xlabel='Epoch',
               ylabel='RMSE test, đo ở mọi epoch')
        ax.legend(fontsize=7.5, loc='upper right')
    fig.suptitle('Tái lập giao thức gốc: 80/20, chuẩn hóa max, lr 1e-3 cố định, chọn epoch theo test', fontsize=11.5)
    fig.tight_layout(); fig.savefig(FIG / 'paper_protocol_curves.png'); plt.close(fig)
    df = pd.DataFrame(rows); df.to_csv(TAB / 'paper_protocol.csv', index=False)
    print(df.round(4).to_string())


if __name__ == '__main__':
    main()
