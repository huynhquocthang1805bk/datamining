"""Assemble reports/BAO_CAO_TGCN.md from the text parts in reports/src/, filling every numeric
table from reports/tables/*.csv so that no number is copied by hand.

Every model-comparison table carries the original baseline (T-GCN) as a marked row, and every
number that is better than the baseline's number in the same column is printed in bold. The
comparison is made at the precision that is displayed (a tie on screen is not bolded).

    python tools/build_report.py && python tools/md2docx.py reports/BAO_CAO_TGCN.md reports/BAO_CAO_TGCN.docx
"""
import json
import re
import subprocess
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from tgcn.plotstyle import label

REP = ROOT / 'reports'; TAB = REP / 'tables'
SRC = REP / 'src'   # hand-written text; numeric tables are filled in from reports/tables/*.csv
PARTS = ['part0.md', 'part_summary_tables.md', 'part1.md', 'part2.md', 'part_critique.md',
         'part3.md', 'part_model.md', 'part4.md', 'part5.md', 'part6.md', 'part8_modern.md', 'part9_end.md']
BASE = ' — *baseline*'                       # marks the reference row (md2docx shades it)
STEP_MIN = {'losloop': (5, 15, 30, 60), 'sztaxi': (15, 30, 45, 60)}


def md(df, index=False):
    cols = ([df.index.name or ''] if index else []) + list(map(str, df.columns))
    out = ['| ' + ' | '.join(cols) + ' |', '|' + '|'.join(['---'] * len(cols)) + '|']
    for i, r in df.iterrows():
        vals = ([str(i)] if index else []) + [str(v) for v in r.values]
        out.append('| ' + ' | '.join(vals) + ' |')
    return '\n'.join(out)


def f3(x):
    return '—' if x is None or (isinstance(x, float) and not np.isfinite(x)) else f'{x:.3f}'


# ---------------------------------------------------------------- bold-if-better helpers
def ok(x):
    try:
        return x is not None and bool(np.isfinite(float(x)))
    except (TypeError, ValueError):
        return False


def bnum(v, base=None, d=3, higher=False, sd=None):
    """v with d decimals; bold when strictly better than base at the displayed precision."""
    if not ok(v):
        return '—'
    s = f'{float(v):.{d}f}'
    if ok(base):
        a, b = round(float(v), d), round(float(base), d)
        if (a > b) if higher else (a < b):
            s = f'**{s}**'
    if ok(sd):
        s += f' ± {float(sd):.{d}f}' if round(float(sd), d) > 0 else f' ± <{10 ** -d:.{d}f}'
    return s


def bcell(vals, bases=None, digits=None, highers=None):
    """'a / b / c' with every component bolded on its own."""
    if not any(ok(v) for v in vals):
        return '—'
    n = len(vals)
    bases = bases if bases is not None else [None] * n
    digits = digits or [2] * n
    highers = highers or [False] * n
    return ' / '.join(bnum(v, b, d, h) for v, b, d, h in zip(vals, bases, digits, highers))


def pnum(s):
    """'7.516 ± 0.031' -> (7.516, 0.031); '8.418' -> (8.418, nan); '—' -> (nan, nan)."""
    if isinstance(s, (int, float, np.integer, np.floating)):
        return float(s), np.nan
    s = str(s).strip()
    if s in ('—', '', 'nan', 'None'):
        return np.nan, np.nan
    if '±' in s:
        a, b = s.split('±')
        return float(a), float(b)
    return float(s), np.nan


def pct(v, b):
    return '—' if not (ok(v) and ok(b)) else f'{100 * (float(v) / float(b) - 1):+.1f}%'


def note(base, acc=False, where='dòng baseline'):
    return (f'*Số in đậm = tốt hơn {base} ({where}) ở cùng cột, so ở đúng số chữ số hiển thị; '
            + ('RMSE, MAE thấp hơn và Accuracy, R² cao hơn là tốt hơn.*' if acc else 'sai số thấp hơn là tốt hơn.*'))


NOTE_TGCN = 'T-GCN gốc'
NOTE_PAPER = 'T-GCN trong Bảng 1 của bài báo'


# ---------------------------------------------------------------- summary tables (vs the original baseline)
KIND = {'paper': 'Bảng 1 bài báo', 'local': 'chạy lại (CPU)', 'ens': 'chạy lại (CPU)', 'gpu': 'chạy lại (GPU của bạn)'}


def _vs(ds):
    return pd.read_csv(TAB / f'vs_baseline_{ds}.csv')


def _seeds(r):
    if r.kind == 'paper':
        return '—'
    if r.kind == 'ens':
        return 'gộp 3 seed'
    return str(int(r.n))


def vs_paper_table(ds):
    df = _vs(ds)
    df = df[(df.summary == 1) & df.c15_RMSE.notna()]
    b = df[df.key == 'paper_tgcn'].iloc[0]
    rows = []
    for _, r in df.iterrows():
        isb = r.key == 'paper_tgcn'
        row = {'Mô hình': r.label + (BASE if isb else ''), 'nguồn': KIND[r.kind]}
        for k in (15, 30, 45, 60):
            cols = [f'c{k}_RMSE', f'c{k}_MAE', f'c{k}_Acc', f'c{k}_R2']
            row[f'"{k} phút"'] = bcell([r[c] for c in cols], None if isb else [b[c] for c in cols],
                                       [3, 3, 3, 3], [False, False, True, True])
        row['Δ "60 phút" RMSE / MAE'] = '—' if isb else f'{pct(r.c60_RMSE, b.c60_RMSE)} / {pct(r.c60_MAE, b.c60_MAE)}'
        rows.append(row)
    return (md(pd.DataFrame(rows)) + '\n\n*Mỗi ô: RMSE / MAE / Accuracy / R², tính như bài báo (sai số gộp các bước 1…k; '
            + ('mọi mục tiêu kể cả số 0, như bài báo' if ds == 'sztaxi' else 'tầm k = 3, 6, 9, 12 bước 5 phút')
            + '). Δ = thay đổi so với baseline (âm là tốt hơn).*\n\n' + note(NOTE_PAPER, acc=True))


def vs_strict_table(ds):
    df = _vs(ds)
    df = df[(df.summary == 1) & (df.kind != 'paper')]
    b = df[df.key == 'tgcn'].iloc[0]
    rows = []
    for _, r in df.iterrows():
        isb = r.key == 'tgcn'
        B = (lambda c: None) if isb else (lambda c, b=b: b[c])
        row = {'Mô hình': r.label + (BASE if isb else ''), 'chạy trên': KIND[r.kind].replace('chạy lại ', '').strip('()'),
               'seed': _seeds(r)}
        if ds == 'sztaxi':
            row['RMSE ≠0'] = bnum(r.RMSE_nz, B('RMSE_nz'), 3, sd=r.RMSE_nz_sd)
            row['MAE ≠0'] = bnum(r.MAE_nz, B('MAE_nz'), 3, sd=r.MAE_nz_sd)
            row['RMSE (mọi mục tiêu)'] = bnum(r.RMSE, B('RMSE'), 3)
            row['MAE (mọi mục tiêu)'] = bnum(r.MAE, B('MAE'), 3)
            for k in (15, 60):
                row[f'{k} phút (mọi mục tiêu)'] = bcell([r[f's{k}_RMSE'], r[f's{k}_MAE']],
                                                        None if isb else [b[f's{k}_RMSE'], b[f's{k}_MAE']])
            row['Δ RMSE / MAE (≠0)'] = '—' if isb else f'{pct(r.RMSE_nz, b.RMSE_nz)} / {pct(r.MAE_nz, b.MAE_nz)}'
        else:
            row['RMSE TB 12 tầm'] = bnum(r.RMSE, B('RMSE'), 3, sd=r.RMSE_sd)
            row['MAE TB 12 tầm'] = bnum(r.MAE, B('MAE'), 3, sd=r.MAE_sd)
            for k in STEP_MIN[ds]:
                row[f'{k} phút'] = bcell([r[f's{k}_RMSE'], r[f's{k}_MAE']], None if isb else [b[f's{k}_RMSE'], b[f's{k}_MAE']])
            row['Δ RMSE / MAE'] = '—' if isb else f'{pct(r.RMSE, b.RMSE)} / {pct(r.MAE, b.MAE)}'
        rows.append(row)
    foot = ('*"15 phút", "60 phút" = RMSE / MAE tại đúng bước đó, trên mọi mục tiêu (lần chạy GPU không xuất sai số từng bước trên '
            'mục tiêu ≠ 0). Δ so với T-GCN gốc, trên mục tiêu ≠ 0.*' if ds == 'sztaxi' else
            '*"5 phút … 60 phút" = RMSE / MAE tại đúng bước đó; ± = độ lệch chuẩn giữa các seed. Δ so với T-GCN gốc (âm là tốt hơn).*')
    return md(pd.DataFrame(rows)) + '\n\n' + foot + '\n\n' + note('T-GCN gốc tái lập, TB 3 seed')


def phase2_table():
    df = _vs('losloop').set_index('key')
    b = df.loc['tgcn']
    rows = []
    for k in ('naive', 'gcn', 'gru', 'tgcn'):
        r = df.loc[k]; isb = k == 'tgcn'
        row = {'Mô hình': ('T-GCN gốc' if isb else r.label) + (BASE if isb else ''), 'seed': _seeds(r),
               'RMSE TB 12 tầm': bnum(r.RMSE, None if isb else b.RMSE, 3, sd=r.RMSE_sd),
               'MAE TB 12 tầm': bnum(r.MAE, None if isb else b.MAE, 3, sd=r.MAE_sd)}
        for m in STEP_MIN['losloop']:
            row[f'{m} phút'] = bcell([r[f's{m}_RMSE'], r[f's{m}_MAE']], None if isb else [b[f's{m}_RMSE'], b[f's{m}_MAE']])
        rows.append(row)
    return (md(pd.DataFrame(rows)) + '\n\n*(ô "5 phút … 60 phút" = RMSE / MAE tại đúng bước đó)*\n\n' + note(NOTE_TGCN))


# ---------------------------------------------------------------- section 5 tables
def peak_table():
    df = pd.read_csv(TAB / 'peak_subsets_losloop.csv')
    order = ['naive', 'linear', 'gru', 'gcn', 'tgcn', 'tgcn_selfsep', 'gru_res_time', 'tgcn_res_time']
    cols = ['tất cả', 'ngoài cao điểm', 'cao điểm', 'thông thoáng', 'kẹt xe (v<40)', 'biến động mạnh (Δ>10)']
    get = lambda m, h, c: (lambda v: float(v.iloc[0]) if len(v) else np.nan)(
        df[(df.model == m) & (df.horizon == h) & (df.subset == c)]['MAE'])

    def block(h):
        rows = []
        for m in order:
            if df[df.model == m].empty:
                continue
            isb = m == 'tgcn'
            r = {'Mô hình': ('T-GCN gốc' if isb else label(m)) + (BASE if isb else '')}
            for c in cols:
                r[c] = bnum(get(m, h, c), None if isb else get('tgcn', h, c), 3)
            rows.append(r)
        return md(pd.DataFrame(rows))
    share = df[(df.model == 'naive') & (df.horizon == '15 phút')].set_index('subset')['share_%']
    tail = ('*Tỉ lệ mẫu test: ' + ', '.join(f'{c} {share.get(c, np.nan):.0f}%' for c in cols[1:]) + '.*')
    return ('**MAE – dự báo trước 15 phút**\n\n' + block('15 phút') + '\n\n**MAE – dự báo trước 60 phút**\n\n'
            + block('60 phút') + '\n\n' + tail + '\n\n' + note(NOTE_TGCN))


def ablation_table():
    df = pd.read_csv(TAB / 'ablation_losloop.csv')
    b = df[df.config == 'T-GCN (gốc)'].iloc[0]
    rows = []
    for _, r in df.iterrows():
        isb = r.config == 'T-GCN (gốc)'
        rows.append({'Cấu hình (seed 42)': r.config + (BASE if isb else ''),
                     'RMSE TB': bnum(r.RMSE_all, None if isb else b.RMSE_all),
                     'MAE TB': bnum(r.MAE_all, None if isb else b.MAE_all),
                     '15 phút': bcell([r.RMSE_15min, r.MAE_15min], None if isb else [b.RMSE_15min, b.MAE_15min]),
                     '60 phút': bcell([r.RMSE_60min, r.MAE_60min], None if isb else [b.RMSE_60min, b.MAE_60min]),
                     'ΔRMSE vs T-GCN': '—' if isb else f"{r['ΔRMSE_all_vs_TGCN_%']:+.1f}%",
                     'ΔMAE vs T-GCN': '—' if isb else f"{r['ΔMAE_all_vs_TGCN_%']:+.1f}%",
                     'epoch chọn': r.best_epoch})
    return md(pd.DataFrame(rows)) + '\n\n' + note('T-GCN gốc cùng seed 42')


def tuning_table():
    df = pd.read_csv(TAB / 'tuning_losloop.csv')
    names = {'T-GCN L=12, h=64 (mặc định)': 'T-GCN, L = 12, hidden = 64 (mặc định)',
             'tgcn_seq6': 'T-GCN, L = 6 (30 phút)', 'tgcn_seq24': 'T-GCN, L = 24 (120 phút)',
             'tgcn_h32': 'T-GCN, hidden = 32', 'tgcn_h100': 'T-GCN, hidden = 100',
             'tgcn_res_time L=12, h=64': 'T-GCN-RT, L = 12', 'tgcn_res_time_seq6': 'T-GCN-RT, L = 6',
             'tgcn_res_time_seq24': 'T-GCN-RT, L = 24'}
    order = list(names)
    df['k'] = df.config.map(lambda c: order.index(c) if c in order else 99)
    df = df.sort_values('k')
    bk = 'T-GCN L=12, h=64 (mặc định)'
    b = df[df.config == bk].iloc[0]
    rows = []
    for _, r in df.iterrows():
        isb = r.config == bk
        rows.append({'Cấu hình (seed 42)': names.get(r.config, r.config) + (BASE if isb else ''),
                     'tham số': '—' if pd.isna(r.n_params) else f'{int(r.n_params):,}',
                     'RMSE TB': bnum(r.RMSE_all, None if isb else b.RMSE_all),
                     'MAE TB': bnum(r.MAE_all, None if isb else b.MAE_all),
                     '15 phút': bcell([r.RMSE_15min, r.MAE_15min], None if isb else [b.RMSE_15min, b.MAE_15min]),
                     '60 phút': bcell([r.RMSE_60min, r.MAE_60min], None if isb else [b.RMSE_60min, b.MAE_60min]),
                     'epoch chọn': r.best_epoch,
                     'thời gian train (phút)': '—' if pd.isna(r.train_s) else f'{r.train_s / 60:.0f}'})
    return md(pd.DataFrame(rows)) + '\n\n' + note('T-GCN gốc mặc định cùng seed 42')


def main_table(ds='losloop', masked=False):
    df = pd.read_csv(TAB / f'main_results_{ds}{"_masked" if masked else ""}.csv')
    keep = [m for m in df.model if not m.startswith('ensemble') or m in
            ('ensemble_gru_res_time', 'ensemble_gru_res_time_rawloss', 'ensemble_mix', 'ensemble_tgcn_selfsep')]
    df = df[df.model.isin(keep)]
    b = df[df.model == 'tgcn'].iloc[0]
    rows = []
    for _, r in df.iterrows():
        isb = r.model == 'tgcn'

        def cell(c, allow=True):
            v, sd = pnum(r[c])
            return bnum(v, None if (isb or not allow) else pnum(b[c])[0], 3, sd=sd)
        rows.append({'Mô hình': ('T-GCN gốc' if isb else label(r.model)) + (BASE if isb else ''), 'seed': r.n_seeds,
                     'RMSE val (chọn)': cell('VAL_RMSE (chọn mô hình)', allow=not r.model.endswith('_masked')),
                     'RMSE TB': cell('overall_RMSE'), 'MAE TB': cell('overall_MAE'),
                     '15 phút RMSE': cell('15min_RMSE'), '15 phút MAE': cell('15min_MAE'),
                     '60 phút RMSE': cell('60min_RMSE'), '60 phút MAE': cell('60min_MAE')})
    return md(pd.DataFrame(rows)) + '\n\n' + note(NOTE_TGCN)


def final_table():
    df = pd.read_csv(TAB / 'main_results_losloop.csv').set_index('model')
    b = df.loc['tgcn']
    VAL = 'VAL_RMSE (chọn mô hình)'

    def cell(r, c, isb):
        v, sd = pnum(r[c])
        return bnum(v, None if isb else pnum(b[c])[0], 3, sd=sd)
    rows = []
    for m in ['gru', 'tgcn', 'tgcn_selfsep', 'gru_res_time', 'tgcn_res_time', 'tgcn_selfsep_res_time', 'gru_res_time_rawloss']:
        if m not in df.index:
            continue
        r = df.loc[m]; isb = m == 'tgcn'
        rows.append({'Cấu hình (3 seed)': ('T-GCN gốc' if isb else label(m)) + (BASE if isb else ''),
                     'RMSE validation ↓ (tiêu chí chọn)': cell(r, VAL, isb),
                     'RMSE test': cell(r, 'overall_RMSE', isb), 'MAE test': cell(r, 'overall_MAE', isb)})
    for m in ('ensemble_gru_res_time', 'ensemble_tgcn_selfsep', 'ensemble_mix'):
        if m in df.index:
            r = df.loc[m]
            rows.append({'Cấu hình (3 seed)': label(m) if m != 'ensemble_tgcn_selfsep' else 'Ensemble T-GCN tách self (3 seed)',
                         'RMSE validation ↓ (tiêu chí chọn)': '—', 'RMSE test': cell(r, 'overall_RMSE', False),
                         'MAE test': cell(r, 'overall_MAE', False)})
    return md(pd.DataFrame(rows)) + '\n\n' + note(NOTE_TGCN)


# ---------------------------------------------------------------- sections 6-8
def fi_table():
    df = pd.read_csv(TAB / 'feature_importance_losloop.csv')
    kinds = ['lịch sử của chính nút', 'láng giềng 1-hop', 'nút ngẫu nhiên xa (≥4 hop)', 'đặc trưng thời gian (giờ, thứ)']
    rows = []
    for m in dict.fromkeys(df.model):
        s = df[df.model == m].set_index('feature')
        lag = df[(df.model == m) & (df.group == 'lag')].set_index('lag')['ΔRMSE']
        r = {'Mô hình': label(m), 'RMSE gốc': f3(s.RMSE_base.iloc[0]),
             'lag 5 phút': f3(lag.get(1.0, np.nan)), 'lag 15 phút': f3(lag.get(3.0, np.nan)),
             'lag 60 phút': f3(lag.get(12.0, np.nan))}
        for k in kinds:
            r[k] = f3(s['ΔRMSE'].get(k, np.nan)) if k in s.index else '—'
        rows.append(r)
    return '*ΔRMSE = RMSE tăng thêm khi xáo trộn nhóm đầu vào (cột không gian: RMSE tại nút đích, TB 40 nút).*\n\n' + md(pd.DataFrame(rows))


def seed_table():
    df = pd.read_csv(TAB / 'rel_seeds_losloop.csv')
    b = df[df.model == 'tgcn'].iloc[0]
    rows = []
    for _, r in df.iterrows():
        isb = r.model == 'tgcn'
        rows.append({'Mô hình': ('T-GCN gốc' if isb else label(r.model)) + (BASE if isb else ''), 'seed': r.seeds,
                     'RMSE': bnum(r.RMSE_mean, None if isb else b.RMSE_mean, 3, sd=r.RMSE_std),
                     'CV RMSE': f"{r['RMSE_CV_%']:.2f}%",
                     'MAE': bnum(r.MAE_mean, None if isb else b.MAE_mean, 3, sd=r.MAE_std),
                     'epoch chọn': r.best_epochs})
    return md(pd.DataFrame(rows)) + '\n\n' + note(NOTE_TGCN) + ' *(Chỉ so cột RMSE và MAE.)*'


def pairwise_table():
    df = pd.read_csv(TAB / 'rel_pairwise_losloop.csv')
    t = pd.DataFrame({'A': df.A.map(label), 'B': df.B.map(label),
                      'RMSE A': df.RMSE_A.map(f3), 'RMSE B': df.RMSE_B.map(f3),
                      'ΔRMSE (A−B)': df['ΔRMSE (A−B)'].map(lambda v: f'{v:+.3f}'),
                      'KTC 95%': [f'[{a:+.3f}; {b:+.3f}]' for a, b in zip(df.CI95_low, df.CI95_high)],
                      'DM p': df.DM_p.map(lambda v: f'{v:.4f}' if v >= 1e-4 else '< 0.0001'),
                      'Kết luận': [('A tốt hơn' if d < 0 else 'B tốt hơn') if p < 0.05 else 'không khác biệt'
                                   for d, p in zip(df['ΔRMSE (A−B)'], df.DM_p)]})
    return md(t)


def conformal_table():
    df = pd.read_csv(TAB / 'rel_conformal_losloop.csv')
    df = df[df['mode'] == 'per_horizon']
    t = pd.DataFrame({'Mô hình': df.model.map(label), 'tầm': df.horizon_min.map(lambda v: f'{v} phút'),
                      'độ phủ tổng (%)': df['coverage_all_%'].map(lambda v: f'{v:.1f}'),
                      'cao điểm (%)': df['coverage_peak_%'].map(lambda v: f'{v:.1f}'),
                      'ngoài cao điểm (%)': df['coverage_offpeak_%'].map(lambda v: f'{v:.1f}'),
                      'độ rộng TB (mph)': df.mean_width.map(lambda v: f'{v:.1f}')})
    return md(t)


# ---------------------------------------------------------------- section 9 (paper)
def paper_table():
    df = pd.read_csv(TAB / 'paper_comparison_losloop.csv')
    base = df[df.model == 'tgcn'].set_index('horizon')
    names = ['ha_window', 'linear', 'gcn', 'gru', 'tgcn', 'naive', 'gru_res_time', 'ensemble_gru_res_time']
    rows = []
    for m in names:
        s = df[df.model == m]
        if s.empty: continue
        for who in ('paper', 'ours'):
            if f'{who}_RMSE' not in s or s[f'{who}_RMSE'].isna().all(): continue
            isb = m == 'tgcn' and who == 'paper'
            r = {'Mô hình': label(m) + (' (SVR)' if m == 'linear' and who == 'paper' else '') + (BASE if isb else ''),
                 'nguồn': 'bài báo' if who == 'paper' else 'chúng tôi'}
            for _, x in s.iterrows():
                bb = base.loc[x.horizon]
                r[x.horizon.replace('min', ' phút')] = bcell(
                    [x[f'{who}_RMSE'], x[f'{who}_MAE'], x[f'{who}_Accuracy'], x[f'{who}_R2']],
                    None if isb else [bb.paper_RMSE, bb.paper_MAE, bb.paper_Accuracy, bb.paper_R2],
                    [3, 3, 3, 3], [False, False, True, True])
            rows.append(r)
    return md(pd.DataFrame(rows)) + '\n\n' + note(NOTE_PAPER, acc=True)


def paper_protocol_table():
    df = pd.read_csv(TAB / 'paper_protocol.csv')
    base = df[df.model == 'tgcn'].set_index('dataset')
    rows = []
    for _, r in df.iterrows():
        bb = base.loc[r.dataset]
        BR, BM, BA, B2 = bb.paper_RMSE, bb.paper_MAE, bb.paper_Accuracy, bb.paper_R2
        isb = r.model == 'tgcn'
        ours = bcell([r.RMSE, r.MAE], [BR, BM], [3, 3])
        if r.model == 'naive':
            paper = '— (không báo cáo)'
        else:
            paper = bcell([r.paper_RMSE, r.paper_MAE], None if isb else [BR, BM], [3, 3]) + (' *(baseline)*' if isb else '')
        nn = r.model not in ('naive', 'ha_window')
        rows.append({'Bộ dữ liệu': r.dataset, 'Mô hình': label(r.model),
                     'epoch train': f'{int(r.epochs)} (tốt nhất: {int(r.best_epoch)})' if nn else '—',
                     'Chúng tôi (RMSE / MAE)': ours, 'Bài báo (RMSE / MAE)': paper,
                     'Accuracy / R² — chúng tôi': bcell([r.Accuracy, r.R2], [BA, B2], [3, 3], [True, True]) if nn else '—',
                     'Accuracy / R² — bài báo': (bcell([r.paper_Accuracy, r.paper_R2], None if isb else [BA, B2], [3, 3],
                                                       [True, True]) + (' *(baseline)*' if isb else '')) if nn else '—'})
    return md(pd.DataFrame(rows)) + '\n\n' + note('T-GCN trong bài báo, cùng bộ dữ liệu, "15 phút"', acc=True,
                                                     where='ô đánh dấu baseline')


# ---------------------------------------------------------------- section 11 (modern models)
def modern_table(ds='losloop'):
    df = pd.read_csv(TAB / f'modern_results_{ds}.csv')
    b = df[df.name == 'tgcn'].iloc[0]
    rows = []
    for _, r in df.iterrows():
        isb = r['name'] == 'tgcn'
        B = (lambda c: None) if isb else (lambda c, b=b: b[c])
        rows.append({'Mô hình': ('T-GCN gốc' if isb else r['Mô hình']) + (BASE if isb else ''), 'nhóm': r['nhóm'],
                     'seed': r.seed, 'tham số': '—' if pd.isna(r['tham số']) else f"{int(r['tham số']):,}",
                     'RMSE val': bnum(r['RMSE val'], B('RMSE val')),
                     'RMSE test': bnum(r.RMSE, B('RMSE'), sd=r.RMSE_sd), 'MAE test': bnum(r.MAE, B('MAE'), sd=r.MAE_sd),
                     '15 phút': bcell([r['RMSE 15'], r['MAE 15']], None if isb else [b['RMSE 15'], b['MAE 15']]),
                     '60 phút': bcell([r['RMSE 60'], r['MAE 60']], None if isb else [b['RMSE 60'], b['MAE 60']]),
                     'phút/run': '—' if pd.isna(r['phút train/run']) else f"{r['phút train/run']:.0f}"})
    head = ('*RMSE / MAE trung bình 12 tầm; "15 phút", "60 phút" = RMSE / MAE tại đúng bước đó; thời gian train trên 1 nhân CPU, có tranh chấp. '
            if ds == 'losloop' else
            '*Mọi sai số tính trên mục tiêu ≠ 0; RMSE / MAE trung bình 4 tầm; "15 phút", "60 phút" = tại đúng bước đó. ')
    gpu = ('Dòng "(GPU)": chạy trên GPU máy của bạn, 3 seed (42, 123, 2026), cùng giao thức; tệp kết quả không có RMSE validation và thời gian'
           + (', cũng không có sai số từng bước trên mục tiêu ≠ 0 (—).*' if ds == 'sztaxi' else '.*'))
    return md(pd.DataFrame(rows)) + '\n\n' + head + gpu + '\n\n' + note(NOTE_TGCN)


def modern_tests(ds='losloop'):
    df = pd.read_csv(TAB / f'modern_tests_{ds}.csv')
    df = df[~df.A.str.startswith('Ensemble GRU-RTP')]
    t = pd.DataFrame({'A': df.A, 'B': df.B, 'ΔRMSE (A−B)': df['ΔRMSE (A−B)'].map(lambda v: f'{v:+.3f}'),
                      'KTC 95%': [f'[{a:+.3f}; {b:+.3f}]' for a, b in zip(df.CI95_low, df.CI95_high)],
                      'DM p': df.DM_p.map(lambda v: f'{v:.4f}' if v >= 1e-4 else '< 0.0001'),
                      'Kết luận': [('A tốt hơn' if d < 0 else 'B tốt hơn') if p < 0.05 else 'không khác biệt'
                                   for d, p in zip(df['ΔRMSE (A−B)'], df.DM_p)]})
    return md(t)


def modern_subsets(ds='losloop'):
    df = pd.read_csv(TAB / f'modern_subsets_{ds}.csv')
    names = {'naive': 'Persistence', 'tgcn': 'T-GCN gốc (dự báo TB 3 seed)' + BASE,
             'gru_res_time': 'GRU-RT (dự báo TB 3 seed)', 'gru_rtp': 'GRU-RTP (dự báo TB 3 seed)',
             'lgbm_best': 'LightGBM (dự báo TB 3 seed)', 'lgbm': 'LightGBM (dự báo TB 3 seed)', 'blend': 'Trộn theo tầm'}
    out = []
    for h in ('15 phút', '60 phút'):
        pv = df[df.horizon == h].pivot_table(index='model', columns='subset', values='MAE', sort=False)
        has_b = 'tgcn' in pv.index
        t = pd.DataFrame({c: [bnum(pv.loc[i, c], pv.loc['tgcn', c] if has_b and i != 'tgcn' else None) for i in pv.index]
                          for c in pv.columns})
        t.index = [names.get(i, i) for i in pv.index]; t.index.name = 'Mô hình'
        out.append(f'**MAE – dự báo trước {h}**\n\n' + md(t, index=True))
    return '\n\n'.join(out) + '\n\n' + note(NOTE_TGCN)


def gpu_table(ds):
    sm = pd.read_csv(TAB / f'modern_gpu_summary_{ds}.csv')
    vs = _vs(ds).set_index('key')
    b = vs.loc['tgcn']
    nz = ds == 'sztaxi'
    rows = []
    mr = pd.read_csv(TAB / f'modern_results_{ds}.csv').set_index('name')
    rb = {'Cấu hình': 'T-GCN gốc (tái lập, CPU)' + BASE, 'tham số': f"{int(mr.loc['tgcn', 'tham số']):,}", 'epoch chọn': '—'}
    if nz:
        rb.update({'RMSE ≠0': bnum(b.RMSE_nz, None, sd=b.RMSE_nz_sd), 'MAE ≠0': bnum(b.MAE_nz, None, sd=b.MAE_nz_sd)})
    rb.update({'RMSE': bnum(b.RMSE, None, sd=b.RMSE_sd), 'MAE': bnum(b.MAE, None, sd=b.MAE_sd),
               '15 phút': bcell([b.s15_RMSE, b.s15_MAE]), '60 phút': bcell([b.s60_RMSE, b.s60_MAE]), 'Δ RMSE': '—'})
    rows.append(rb)
    for _, r in sm.iterrows():
        x = {'Cấu hình': r['Mô hình (GPU)'], 'tham số': f"{int(r['tham số']):,}", 'epoch chọn': r['epoch chọn']}
        if nz:
            x.update({'RMSE ≠0': bnum(r['RMSE ≠0'], b.RMSE_nz, sd=r['RMSE ≠0_sd']),
                      'MAE ≠0': bnum(r['MAE ≠0'], b.MAE_nz, sd=r['MAE ≠0_sd'])})
        x.update({'RMSE': bnum(r.RMSE, b.RMSE, sd=r.RMSE_sd), 'MAE': bnum(r.MAE, b.MAE, sd=r.MAE_sd),
                  '15 phút': bcell([r['RMSE 15'], r['MAE 15']], [b.s15_RMSE, b.s15_MAE]),
                  '60 phút': bcell([r['RMSE 60'], r['MAE 60']], [b.s60_RMSE, b.s60_MAE]),
                  'Δ RMSE': pct(r['RMSE ≠0'], b.RMSE_nz) if nz else pct(r.RMSE, b.RMSE)})
        rows.append(x)
    foot = ('*RMSE, MAE: trung bình 4 tầm (cột "≠0": chỉ mục tiêu khác 0, tiêu chí chính của SZ-taxi; hai cột kế: mọi mục tiêu); '
            '"15 phút", "60 phút" = RMSE / MAE tại đúng bước đó, mọi mục tiêu; Δ RMSE trên mục tiêu ≠ 0.*' if nz else
            '*RMSE, MAE: trung bình 12 tầm ± độ lệch chuẩn 3 seed; "15 phút", "60 phút" = RMSE / MAE tại đúng bước đó.*')
    return md(pd.DataFrame(rows)) + '\n\n' + foot + '\n\n' + note('T-GCN gốc tái lập, TB 3 seed')


def gpu_checks():
    ck = pd.read_csv(TAB / 'modern_gpu_checks.csv')
    ck['dataset'] = ck.dataset.map({'losloop': 'Los-loop', 'sztaxi': 'SZ-taxi'})
    keep = ck[~ck['kiểm tra'].str.startswith('tóm tắt của bạn')]
    agree = ck[ck['kiểm tra'].str.startswith('tóm tắt của bạn')]
    rows = keep.rename(columns={'dataset': 'Bộ dữ liệu', 'kiểm tra': 'Kiểm tra', 'kết quả': 'Kết quả'})
    if len(agree):
        n_ok = int((agree['kết quả'] == 'khớp').sum())
        rows = pd.concat([pd.DataFrame([{'Bộ dữ liệu': 'SZ-taxi', 'Kiểm tra': 'tệp tóm tắt của bạn = tính lại từ 21 run (TB và độ lệch chuẩn ddof = 1, 4 chỉ số)',
                                         'Kết quả': f'{n_ok}/{len(agree)} cấu hình khớp tuyệt đối'}]), rows], ignore_index=True)
    return md(rows)


def lgbm_groups(ds='losloop'):
    g = pd.read_csv(TAB / f'lgbm_importance_groups_{ds}.csv', index_col=0)
    g = g.loc[[i for i in g.index if i in (5, 15, 30, 60, 45)]] if ds == 'losloop' else g
    g.index = [f'{i} phút' for i in g.index]; g.index.name = 'Tầm'
    return md((100 * g).round(1).astype(str).apply(lambda c: c + '%'), index=True)


def lgbm_tuning(ds='losloop'):
    return md(pd.read_csv(TAB / f'lgbm_tuning_{ds}.csv'))


def blend_weights(ds='losloop'):
    w = pd.read_csv(TAB / f'blend_weights_{ds}.csv', index_col=0)
    if len(w) > 4:
        w = w.loc[[i for i in w.index if i in ('5 phút', '15 phút', '30 phút', '45 phút', '60 phút')]]
    w.index.name = 'Tầm'
    return md(w.round(2), index=True)


def paper_style_blend():
    m = json.loads((ROOT / 'results/losloop/ensemble/blend/metrics.json').read_text())
    c15, c60 = m['cumulative']['15min'], m['cumulative']['60min']
    return (f"\"15 phút\" RMSE {c15['RMSE']:.2f} / MAE {c15['MAE']:.2f} / Accuracy {c15['Accuracy']:.3f} / R² {c15['R2']:.3f} "
            f"và \"60 phút\" {c60['RMSE']:.2f} / {c60['MAE']:.2f} / {c60['Accuracy']:.3f} / {c60['R2']:.3f} — so với T-GCN "
            f"trong bài báo 5.13 / 3.18 / 0.913 / 0.863 và 7.27 / 4.60 / 0.876 / 0.728 (giảm "
            f"{100 * (1 - c15['RMSE'] / 5.1264):.0f}% và {100 * (1 - c60['RMSE'] / 7.2677):.0f}% RMSE, "
            f"{100 * (1 - c15['MAE'] / 3.1802):.0f}% và {100 * (1 - c60['MAE'] / 4.6021):.0f}% MAE), dù giao thức chặt hơn")


# ---------------------------------------------------------------- loading examples
def model_card():
    df = pd.read_csv(TAB / 'model_card_tgcn_res_time_losloop.csv')
    t = pd.DataFrame({'Tensor': df.tensor.map(lambda s: f'`{s}`'), 'Kích thước': df['shape'],
                      'Số phần tử': df.n.map(lambda v: f'{v:,}'), 'Học?': df.learned.map(lambda v: 'có' if v else 'không'),
                      'Vai trò': df.role})
    return md(t) + f"\n\nTổng số tham số học được: **{int(df[df.learned].n.sum()):,}**."


def _predict(args):
    out = subprocess.run([sys.executable, str(ROOT / 'predict.py'), *args, '--input', str(ROOT / 'data/losloop/speed.csv'),
                          '--end-time', '2012-03-07 07:30', '--compare', '--sensors', '4'], capture_output=True, text=True,
                         cwd=ROOT, env={**__import__('os').environ, 'TGCN_THREADS': '1'}).stdout
    return '```\n' + out.strip() + '\n```'


def predict_example():
    cks = sorted((ROOT / 'models').glob('losloop_gru_res_time_seed*.pt'))
    if not cks:
        return '*(chạy `python tools/export_models.py` trước)*'
    return _predict(['--checkpoint', *map(str, cks)])


def predict_gbdt_example():
    d = ROOT / 'models' / 'losloop_lgbm'
    if not d.is_dir():
        return '*(chạy `python tools/export_models.py` trước)*'
    return _predict(['--checkpoint', str(d)])


def oracle_table():
    df = pd.read_csv(TAB / 'oracle_bias_losloop.csv')
    df['name'] = df.config.str.split('/').str[-1]
    g = df.groupby('name').agg(n=('seed', 'count'), val=('test_RMSE_val_selected', 'mean'),
                               test=('test_RMSE_test_selected', 'mean'), bias=('optimistic_bias_%', 'mean'),
                               bmax=('optimistic_bias_%', 'max'), ev=('best_epoch_val', 'mean'),
                               eo=('oracle_epoch_test', 'mean')).reset_index()
    g = g[~g.name.str.startswith('tgcn_seq') & ~g.name.str.startswith('tgcn_h') &
          ~g.name.str.startswith('tgcn_res_time_seq')].sort_values('bias').reset_index(drop=True)
    t = pd.DataFrame({'Mô hình': g.name.map(label), 'run': g.n,
                      'RMSE test – chọn theo val': g.val.map(f3), 'RMSE test – chọn theo test': g.test.map(f3),
                      'độ lạc quan TB': g.bias.map(lambda v: f'{v:.2f}%'), 'tối đa': g.bmax.map(lambda v: f'{v:.2f}%'),
                      'epoch (val / test)': [f'{a:.0f} / {b:.0f}' for a, b in zip(g.ev, g.eo)]})
    return md(t)


def oracle_stats():
    df = pd.read_csv(TAB / 'oracle_bias_losloop.csv')
    return df['optimistic_bias_%'].mean(), df['optimistic_bias_%'].max()


FILL = {'VS_PAPER_LL': lambda: vs_paper_table('losloop'), 'VS_STRICT_LL': lambda: vs_strict_table('losloop'),
        'VS_PAPER_SZ': lambda: vs_paper_table('sztaxi'), 'VS_STRICT_SZ': lambda: vs_strict_table('sztaxi'),
        'PHASE2_TABLE': phase2_table, 'GPU_TABLE_LL': lambda: gpu_table('losloop'), 'GPU_TABLE_SZ': lambda: gpu_table('sztaxi'),
        'GPU_CHECKS': gpu_checks,
        'MODERN_TABLE': modern_table, 'MODERN_TESTS': modern_tests, 'MODERN_SUBSETS': modern_subsets,
        'LGBM_GROUPS': lgbm_groups, 'LGBM_TUNING': lgbm_tuning, 'BLEND_WEIGHTS': blend_weights,
        'PREDICT_GBDT_EXAMPLE': predict_gbdt_example,
        'SZ_MODERN_TABLE': lambda: modern_table('sztaxi'),
        'FINAL_TABLE': final_table, 'ORACLE_TABLE': oracle_table, 'PAPER_PROTOCOL_TABLE': paper_protocol_table,
        'PEAK_TABLE': peak_table, 'ABLATION_TABLE': ablation_table, 'TUNING_TABLE': tuning_table,
        'MAIN_TABLE': main_table, 'FI_TABLE': fi_table, 'SEED_TABLE': seed_table,
        'PAIRWISE_TABLE': pairwise_table, 'CONFORMAL_TABLE': conformal_table, 'PAPER_TABLE': paper_table,
        'MODEL_CARD_TABLE': model_card, 'PREDICT_EXAMPLE': predict_example,
        'SZ_MAIN_TABLE': lambda: main_table('sztaxi'), 'SZ_MASKED_TABLE': lambda: main_table('sztaxi', True)}


def main():
    text = '\n\n'.join((SRC / p).read_text(encoding='utf-8') for p in PARTS if (SRC / p).is_file())
    for key, fn in FILL.items():
        if re.search(rf'^{key}$', text, flags=re.M):
            try:
                text = re.sub(rf'^{key}$', lambda _: fn(), text, flags=re.M)
            except FileNotFoundError as e:
                print('missing table for', key, e)
    if 'PAPER_STYLE_BLEND' in text:
        try:
            text = text.replace('PAPER_STYLE_BLEND', paper_style_blend())
        except FileNotFoundError:
            pass
    try:
        mean_b, max_b = oracle_stats()
        text = text.replace('ORACLE_MEAN%', f'{mean_b:.2f}%').replace('ORACLE_MAX%', f'{max_b:.2f}%')
    except FileNotFoundError:
        pass
    left = sorted(set(re.findall(r'^[A-Z_]{6,}$', text, flags=re.M)) | set(re.findall(r'(?<![\w/])[A-Z]{2,}_[A-Z_]{3,}(?![\w.])', text)))
    if left:
        print('UNFILLED placeholders:', left)
    (REP / 'BAO_CAO_TGCN.md').write_text(text, encoding='utf-8')
    print('wrote', REP / 'BAO_CAO_TGCN.md', len(text.splitlines()), 'lines')


if __name__ == '__main__':
    main()
