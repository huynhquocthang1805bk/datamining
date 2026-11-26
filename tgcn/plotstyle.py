"""Shared matplotlib style for every figure of the report.

Categorical colours follow the entity (a model keeps its colour in every chart);
sequential maps use a single blue ramp; grid and axes are recessive.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK2 = '#52514e'
GRID = '#e4e3df'
NEUTRAL = '#8a8985'
NEUTRAL_LIGHT = '#bdbcb6'

SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']

MODEL_COLORS = {
    'naive': NEUTRAL, 'ha_window': NEUTRAL_LIGHT, 'ha_tod': '#d6d5cf', 'linear': '#eda100',
    'gru': '#2a78d6', 'gcn': '#1baf7a', 'tgcn': '#eb6834',
    'tgcn_res': '#e87ba4', 'tgcn_time': '#008300', 'tgcn_res_time': '#6250d6',
    'gru_res_time': '#86b6ef', 'tgcn_plus': '#e34948', 'tgcn_res_time_huber': '#9085e9',
    'ensemble': '#0d366b', 'ensemble_mix': '#5598e7', 'ensemble_gru_res_time': '#184f95',
    'tgcn_selfsep_res_time': '#1c5cab', 'tgcn_selfsep': '#0d366b', 'tgcn_selfloop1': '#d6d5cf',
    'gru_res_time_rawloss': '#86b6ef', 'tgcn_res_time_rawloss': '#9085e9',
}
MODEL_LABELS = {
    'naive': 'Naive (giữ giá trị cuối)', 'ha_window': 'HA (TB cửa sổ)', 'ha_tod': 'HA theo giờ',
    'linear': 'Hồi quy tuyến tính', 'gru': 'GRU', 'gcn': 'GCN', 'tgcn': 'T-GCN',
    'tgcn_res': 'T-GCN + residual', 'tgcn_time': 'T-GCN + thời gian',
    'tgcn_res_time': 'T-GCN-RT (residual + thời gian)', 'gru_res_time': 'GRU + residual + thời gian',
    'tgcn_plus': 'T-GCN+ (RT + đồ thị thích nghi)', 'tgcn_res_time_huber': 'T-GCN-RT + Huber',
    'tgcn_res_time_masked': 'T-GCN-RT (loss bỏ qua 0)', 'tgcn_masked': 'T-GCN (loss bỏ qua 0)',
    'ensemble': 'Ensemble T-GCN-RT (3 seed)',
    'tgcn_selfsep': 'T-GCN tách self/láng giềng', 'tgcn_selfsep_res_time': 'T-GCN-RT tách self/láng giềng (T-GCN-RTS)',
    'tgcn_selfloop1': 'T-GCN (self-loop = 1)', 'ensemble_mix': 'Ensemble GRU-RT + T-GCN-RTS',
    'ensemble_gru_res_time': 'Ensemble GRU-RT (3 seed)', 'ensemble_tgcn_res_time': 'Ensemble T-GCN-RT (3 seed)',
    'gru_res_time_rawloss': 'GRU-RT, loss đơn vị gốc', 'tgcn_res_time_rawloss': 'T-GCN-RT, loss đơn vị gốc',
    'gru_res_time': 'GRU-RT (GRU + residual + thời gian)',
    'tgcn_res_time_nograph': 'T-GCN-RT bỏ đồ thị (Â = I)',
}
BLUES = LinearSegmentedColormap.from_list(
    'blues', ['#cde2fb', '#86b6ef', '#3987e5', '#256abf', '#184f95', '#0d366b'])


def color(name, i=0):
    return MODEL_COLORS.get(name, SERIES[i % len(SERIES)])


def label(name):
    return MODEL_LABELS.get(name, name)


def apply():
    plt.rcParams.update({
        'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
        'axes.edgecolor': NEUTRAL_LIGHT, 'axes.labelcolor': INK2, 'axes.titlecolor': INK,
        'axes.titlesize': 11.5, 'axes.titleweight': 'bold', 'axes.labelsize': 9.5,
        'xtick.color': INK2, 'ytick.color': INK2, 'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5,
        'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.7,
        'axes.spines.top': False, 'axes.spines.right': False,
        'legend.frameon': False, 'legend.fontsize': 8.5, 'lines.linewidth': 2.0,
        'font.family': 'DejaVu Sans', 'figure.dpi': 110, 'savefig.dpi': 160,
        'savefig.bbox': 'tight',
    })


apply()
