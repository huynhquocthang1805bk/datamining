"""Modern spatio-temporal forecasters, re-implemented in the same interface as
tgcn/models.py so that they are trained and evaluated exactly like GRU/T-GCN:

    forward(x [B, L, N, 1] z-scored speed, time_feats [B, L, 4]) -> [B, H, N]

Implemented (compact but faithful versions; references in the report):
  * STID        Shao et al., CIKM 2022  — MLP + spatial/temporal identity embeddings
  * GWNet       Wu et al., IJCAI 2019   — Graph WaveNet: gated dilated TCN + diffusion
                                          GCN with forward/backward + adaptive adjacency
  * DCRNN-lite  Li et al., ICLR 2018    — GRU whose gates use K-step bidirectional
                                          diffusion convolution (incl. the identity term);
                                          direct multi-step head instead of seq2seq
  * STAEformer  Liu et al., CIKM 2023   — vanilla Transformer along time and along nodes
                                          with a spatio-temporal adaptive embedding

All take the option residual=True (predict the change w.r.t. the last observed
value, the trick that helped GRU/T-GCN on Los-loop).
"""
import math
import torch
from torch import nn
import torch.nn.functional as F


# ----------------------------------------------------------------- helpers
def calendar_index(time_feats, steps_per_day):
    """Recover integer time-of-day slot and day-of-week from the sin/cos features."""
    tod = torch.atan2(time_feats[..., 0], time_feats[..., 1]) / (2 * math.pi) % 1.0
    dow = torch.atan2(time_feats[..., 2], time_feats[..., 3]) / (2 * math.pi) % 1.0
    tod_i = torch.round(tod * steps_per_day).long() % steps_per_day
    dow_i = torch.round(dow * 7).long() % 7
    return tod_i, dow_i                                   # [B, L] each


def transition_matrices(adj_norm):
    """Random-walk transitions from the (authors') normalised adjacency without
    self-loops: forward P_f = rownorm(A), backward P_b = rownorm(A^T)."""
    a = adj_norm.clone()
    a.fill_diagonal_(0.0)
    def rownorm(m):
        d = m.sum(1, keepdim=True)
        return torch.where(d > 0, m / d.clamp_min(1e-8), torch.zeros_like(m))
    return rownorm(a), rownorm(a.T.contiguous())


def _residual(pred, x, use):
    return pred + x[:, -1, :, 0].unsqueeze(1) if use else pred


# ----------------------------------------------------------------- STID
class STID(nn.Module):
    """Spatial-Temporal IDentity (Shao et al., 2022).

    hidden_i = [W_ts x_i(t-L+1..t), E_node[i], E_tod[slot(t)], E_dow[day(t)]]
    -> 3 residual MLP layers -> H outputs. No graph, no recurrence: the node
    embedding tells the model *which* sensor it is, the time embeddings *when*.
    """

    def __init__(self, num_nodes, seq_len, horizon, steps_per_day, embed_dim=32,
                 node_dim=32, tod_dim=32, dow_dim=32, num_layer=3, dropout=0.15, residual=False):
        super().__init__()
        self.steps_per_day = steps_per_day
        self.use_residual = residual
        self.ts_emb = nn.Linear(seq_len, embed_dim)
        self.node_emb = nn.Parameter(torch.empty(num_nodes, node_dim))
        self.tod_emb = nn.Parameter(torch.empty(steps_per_day, tod_dim))
        self.dow_emb = nn.Parameter(torch.empty(7, dow_dim))
        for p in (self.node_emb, self.tod_emb, self.dow_emb):
            nn.init.xavier_uniform_(p)
        hid = embed_dim + node_dim + tod_dim + dow_dim
        self.layers = nn.ModuleList(nn.Sequential(nn.Linear(hid, hid), nn.ReLU(), nn.Dropout(dropout),
                                                  nn.Linear(hid, hid)) for _ in range(num_layer))
        self.out = nn.Linear(hid, horizon)

    def forward(self, x, time_feats=None):
        b, l, n, _ = x.shape
        tod, dow = calendar_index(time_feats[:, -1], self.steps_per_day)      # [B]
        h = torch.cat([self.ts_emb(x[..., 0].transpose(1, 2)),
                       self.node_emb.unsqueeze(0).expand(b, -1, -1),
                       self.tod_emb[tod].unsqueeze(1).expand(-1, n, -1),
                       self.dow_emb[dow].unsqueeze(1).expand(-1, n, -1)], dim=-1)   # [B,N,hid]
        for layer in self.layers:
            h = h + layer(h)
        return _residual(self.out(h).transpose(1, 2), x, self.use_residual)


# ----------------------------------------------------------------- Graph WaveNet
class _DiffusionGCN(nn.Module):
    """X W_0 + sum over supports A and k=1..order of (A^k X) W_{A,k}  (one Linear on the concat)."""

    def __init__(self, c_in, c_out, n_supports, order=2, dropout=0.3):
        super().__init__()
        self.order = order
        self.mlp = nn.Linear(c_in * (1 + n_supports * order), c_out)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, supports):                      # x [B,N,T,C] (channel-last)
        b, n, t, c = x.shape
        out = [x]
        for a in supports:
            h = x.reshape(b, n, t * c)
            for _ in range(self.order):
                h = torch.matmul(a, h)                   # h_w = sum_v a[w,v] h_v
                out.append(h.reshape(b, n, t, c))
        return self.drop(self.mlp(torch.cat(out, dim=-1)))


class _DilatedConv(nn.Module):
    """Causal temporal convolution, kernel 2, dilation d, written as two Linears
    (identical to Conv2d((1,2), dilation=(1,d)) but much faster on CPU)."""

    def __init__(self, c_in, c_out, d):
        super().__init__()
        self.d = d
        self.w0 = nn.Linear(c_in, c_out)
        self.w1 = nn.Linear(c_in, c_out, bias=False)

    def forward(self, x):                                # x [B,N,T,C] -> [B,N,T-d,C']
        return self.w0(x[:, :, :-self.d]) + self.w1(x[:, :, self.d:])


class GWNet(nn.Module):
    """Graph WaveNet (Wu et al., 2019): 4 blocks x 2 layers of gated dilated TCN
    (dilation 1,2) + diffusion GCN over [P_f, P_b, A_adaptive], skip connections.
    Channel-last implementation; compact channel sizes for CPU by default."""

    def __init__(self, num_nodes, adj_norm, seq_len, horizon, in_dim=5, residual_ch=32, dilation_ch=32,
                 skip_ch=128, end_ch=256, blocks=4, layers=2, adaptive=True, emb_dim=10,
                 dropout=0.3, residual=False):
        super().__init__()
        p_f, p_b = transition_matrices(adj_norm)
        self.register_buffer('p_f', p_f); self.register_buffer('p_b', p_b)
        self.adaptive = adaptive
        if adaptive:
            self.e1 = nn.Parameter(torch.randn(num_nodes, emb_dim) * 0.1)
            self.e2 = nn.Parameter(torch.randn(emb_dim, num_nodes) * 0.1)
        n_sup = 2 + int(adaptive)
        self.use_residual = residual
        self.start = nn.Linear(in_dim, residual_ch)
        self.filt, self.gate, self.skip, self.gconv, self.bn = (nn.ModuleList() for _ in range(5))
        self.dilations = []
        rf = 1
        for _ in range(blocks):
            d = 1
            for _ in range(layers):
                self.dilations.append(d)
                self.filt.append(_DilatedConv(residual_ch, dilation_ch, d))
                self.gate.append(_DilatedConv(residual_ch, dilation_ch, d))
                self.skip.append(nn.Linear(dilation_ch, skip_ch))
                self.gconv.append(_DiffusionGCN(dilation_ch, residual_ch, n_sup, 2, dropout))
                self.bn.append(nn.BatchNorm1d(residual_ch))
                rf += d
                d *= 2
        self.receptive_field = rf
        self.end1 = nn.Linear(skip_ch, end_ch)
        self.end2 = nn.Linear(end_ch, horizon)

    def forward(self, x, time_feats=None):
        b, l, n, _ = x.shape
        inp = torch.cat([x, time_feats.unsqueeze(2).expand(-1, -1, n, -1)], dim=-1)   # [B,L,N,5]
        inp = inp.transpose(1, 2)                                                      # [B,N,L,5]
        if l < self.receptive_field:
            inp = F.pad(inp, (0, 0, self.receptive_field - l, 0))
        h = self.start(inp)
        supports = [self.p_f, self.p_b]
        if self.adaptive:
            supports.append(torch.softmax(torch.relu(self.e1 @ self.e2), dim=1))
        skip = None
        for i in range(len(self.dilations)):
            res = h
            g = torch.tanh(self.filt[i](h)) * torch.sigmoid(self.gate[i](h))
            s = self.skip[i](g)
            skip = s if skip is None else skip[:, :, -s.size(2):] + s
            h = self.gconv[i](g, supports) + res[:, :, -g.size(2):]
            bb, nn_, tt, cc = h.shape
            h = self.bn[i](h.reshape(-1, cc)).reshape(bb, nn_, tt, cc)
        out = self.end2(torch.relu(self.end1(torch.relu(skip[:, :, -1]))))          # [B,N,H]
        return _residual(out.transpose(1, 2), x, self.use_residual)


# ----------------------------------------------------------------- DCRNN-lite
class DiffusionLinear(nn.Module):
    """[X, P_f X, ..., P_f^K X, P_b X, ..., P_b^K X] W + b  (Li et al., 2018).

    Unlike the T-GCN layer, the node's own signal (k = 0) keeps its own weights."""

    def __init__(self, c_in, c_out, k=2, bias_init=0.0):
        super().__init__()
        self.k = k
        self.linear = nn.Linear(c_in * (1 + 2 * k), c_out)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.constant_(self.linear.bias, bias_init)

    def forward(self, x, supports):                     # x [B,N,C]
        out = [x]
        for p in supports:
            h = x
            for _ in range(self.k):
                h = torch.matmul(p, h)
                out.append(h)
        return self.linear(torch.cat(out, dim=-1))


class DCRNNLite(nn.Module):
    def __init__(self, num_nodes, adj_norm, horizon, hidden_dim=64, k=2, time_features=True,
                 residual=False, dropout=0.0):
        super().__init__()
        p_f, p_b = transition_matrices(adj_norm)
        self.register_buffer('p_f', p_f); self.register_buffer('p_b', p_b)
        self.use_time = time_features; self.use_residual = residual
        c = 1 + (4 if time_features else 0) + hidden_dim
        self.hidden_dim = hidden_dim
        self.gates = DiffusionLinear(c, 2 * hidden_dim, k, bias_init=1.0)
        self.cand = DiffusionLinear(c, hidden_dim, k)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, horizon)

    def forward(self, x, time_feats=None):
        b, l, n, _ = x.shape
        sup = (self.p_f, self.p_b)
        h = x.new_zeros(b, n, self.hidden_dim)
        for t in range(l):
            xt = x[:, t]
            if self.use_time:
                xt = torch.cat([xt, time_feats[:, t].unsqueeze(1).expand(-1, n, -1)], dim=-1)
            r, u = torch.sigmoid(self.gates(torch.cat([xt, h], -1), sup)).chunk(2, -1)
            c = torch.tanh(self.cand(torch.cat([xt, r * h], -1), sup))
            h = u * h + (1 - u) * c
        return _residual(self.head(self.dropout(h)).transpose(1, 2), x, self.use_residual)


# ----------------------------------------------------------------- STAEformer
class _AttnLayer(nn.Module):
    def __init__(self, d, ff, heads, dropout):
        super().__init__()
        # no dropout on the attention weights (as in the official STAEformer); masks of
        # size [B*L, heads, N, N] dominated the CPU time
        self.attn = nn.MultiheadAttention(d, heads, dropout=0.0, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.ReLU(), nn.Linear(ff, d))
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):                                # x [*, S, d] attention over S
        x = self.n1(x + self.drop(self.attn(x, x, x, need_weights=False)[0]))
        return self.n2(x + self.drop(self.ff(x)))


class STAEformer(nn.Module):
    """Liu et al., 2023: input / time-of-day / day-of-week / spatio-temporal adaptive
    embeddings -> Transformer layers along time, then along nodes -> mixed projection."""

    def __init__(self, num_nodes, seq_len, horizon, steps_per_day, input_dim=24, tod_dim=24,
                 dow_dim=24, adaptive_dim=80, ff_dim=256, heads=4, layers=3, dropout=0.1,
                 residual=False):
        super().__init__()
        self.steps_per_day = steps_per_day; self.use_residual = residual
        self.seq_len = seq_len
        self.inp = nn.Linear(1, input_dim)
        self.tod = nn.Embedding(steps_per_day, tod_dim)
        self.dow = nn.Embedding(7, dow_dim)
        self.adp = nn.Parameter(torch.empty(seq_len, num_nodes, adaptive_dim))
        nn.init.xavier_uniform_(self.adp)
        d = input_dim + tod_dim + dow_dim + adaptive_dim
        self.t_layers = nn.ModuleList(_AttnLayer(d, ff_dim, heads, dropout) for _ in range(layers))
        self.s_layers = nn.ModuleList(_AttnLayer(d, ff_dim, heads, dropout) for _ in range(layers))
        self.out = nn.Linear(seq_len * d, horizon)

    def forward(self, x, time_feats=None):
        b, l, n, _ = x.shape
        tod, dow = calendar_index(time_feats, self.steps_per_day)                 # [B,L]
        h = torch.cat([self.inp(x),
                       self.tod(tod).unsqueeze(2).expand(-1, -1, n, -1),
                       self.dow(dow).unsqueeze(2).expand(-1, -1, n, -1),
                       self.adp.unsqueeze(0).expand(b, -1, -1, -1)], dim=-1)       # [B,L,N,d]
        d = h.shape[-1]
        h = h.transpose(1, 2).reshape(b * n, l, d)                                 # time attention
        for layer in self.t_layers:
            h = layer(h)
        h = h.reshape(b, n, l, d).transpose(1, 2).reshape(b * l, n, d)            # node attention
        for layer in self.s_layers:
            h = layer(h)
        h = h.reshape(b, l, n, d).permute(0, 2, 1, 3).reshape(b, n, l * d)
        return _residual(self.out(h).transpose(1, 2), x, self.use_residual)


MODERN = ('stid', 'gwnet', 'dcrnn', 'staeformer')


def make_modern(name, bundle, seq_len, horizon, hidden_dim=64, residual=False, size='compact',
                dropout=None):
    """size='compact' fits a 4-core CPU; size='full' = the papers' default sizes (GPU)."""
    n = len(bundle.sensor_ids)
    adj = torch.as_tensor(bundle.adj, dtype=torch.float32)
    spd = int(round(1440 / bundle.interval_min))
    full = size == 'full'
    if name == 'stid':
        return STID(n, seq_len, horizon, spd, residual=residual,
                    dropout=0.15 if dropout is None else dropout)
    if name == 'gwnet':
        kw = dict(residual_ch=32, dilation_ch=32, skip_ch=256, end_ch=512) if full else \
             dict(residual_ch=32, dilation_ch=32, skip_ch=128, end_ch=256)
        return GWNet(n, adj, seq_len, horizon, residual=residual,
                     dropout=0.3 if dropout is None else dropout, **kw)
    if name == 'dcrnn':
        return DCRNNLite(n, adj, horizon, hidden_dim=hidden_dim, k=2, residual=residual,
                         dropout=0.0 if dropout is None else dropout)
    if name == 'staeformer':
        kw = dict(input_dim=24, tod_dim=24, dow_dim=24, adaptive_dim=80, ff_dim=256, layers=3) if full else \
             dict(input_dim=16, tod_dim=16, dow_dim=16, adaptive_dim=32, ff_dim=128, layers=2)
        return STAEformer(n, seq_len, horizon, spd, residual=residual,
                          dropout=0.1 if dropout is None else dropout, **kw)
    raise ValueError(name)
