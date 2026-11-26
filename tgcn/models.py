"""GRU, GCN, T-GCN and T-GCN+ for multi-node traffic speed forecasting.

Independent PyTorch implementation motivated by Zhao et al. (2019). Shapes:
    x          [B, L, N, 1]   standardized speed of the last L steps
    time_feats [B, L, 4]      sin/cos(time-of-day), sin/cos(day-of-week)
    output     [B, H, N]      standardized speed of the next H steps

Options shared by GRU and T-GCN (so ablations are comparable):
    time_features : append the 4 calendar features to every node input.
    residual      : predict y_{t+k} = x_t + delta_k (learn the *change* from the
                    last observation instead of the absolute level).
T-GCN only:
    adaptive_graph: A_used = (1-a) * A_phys + a * softmax(relu(E E^T)).
"""
import numpy as np
import torch
from torch import nn


class GraphLinear(nn.Module):
    """GCN layer used inside the GRU gates: f(A, X) = A X W + b.

    self_sep=True: f(A, X) = [X, A X] W + b, i.e. separate weights for the node
    itself and for its neighbourhood (the k=0 term of DCRNN's diffusion
    convolution). With the symmetric normalisation of T-GCN a node keeps only
    ~20% of its own signal in A X on Los-loop; this variant keeps 100%.
    """

    def __init__(self, input_dim, output_dim, bias_init=0.0, self_sep=False):
        super().__init__()
        self.self_sep = bool(self_sep)
        self.linear = nn.Linear(input_dim * (2 if self_sep else 1), output_dim)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.constant_(self.linear.bias, bias_init)

    def forward(self, x, adj):
        # x: [B,N,C]; adjacency: [N,N]  ->  aggregate neighbours, then project
        ax = graph_matmul(adj, x)
        return self.linear(torch.cat((x, ax), dim=-1) if self.self_sep else ax)


def graph_matmul(adj, x):
    """A @ x for x [B,N,C]; uses a sparse kernel when adj is sparse (6.6% non-zeros
    on Los-loop, 2.2% on SZ-taxi -> ~10x faster on CPU than a dense matmul)."""
    if adj.is_sparse:
        b, n, c = x.shape
        y = torch.sparse.mm(adj, x.permute(1, 0, 2).reshape(n, b * c))
        return y.reshape(n, b, c).permute(1, 0, 2)
    return torch.matmul(adj, x)


class TGCNCell(nn.Module):
    """GRU cell whose linear maps are graph convolutions (Eq. 3-6 of the paper)."""

    def __init__(self, input_dim, hidden_dim, self_sep=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        # bias 1.0 on gates as in the authors' code: start with u≈0.73 (keep memory)
        self.gates = GraphLinear(input_dim + hidden_dim, 2 * hidden_dim, bias_init=1.0, self_sep=self_sep)
        self.candidate = GraphLinear(input_dim + hidden_dim, hidden_dim, self_sep=self_sep)

    def forward(self, x, h, adj, return_gates=False):
        r, u = torch.sigmoid(self.gates(torch.cat((x, h), dim=-1), adj)).chunk(2, -1)
        c = torch.tanh(self.candidate(torch.cat((x, r * h), dim=-1), adj))
        h_new = u * h + (1 - u) * c
        if return_gates:
            return h_new, (r, u)
        return h_new


class AdaptiveGraph(nn.Module):
    """Learn soft adjacency from node embeddings; mixture with physical graph."""

    def __init__(self, num_nodes, embed_dim=8, alpha_init=-1.4):
        super().__init__()
        self.emb = nn.Parameter(0.1 * torch.randn(num_nodes, embed_dim))
        self.alpha_logit = nn.Parameter(torch.tensor(alpha_init))

    def learned(self):
        return torch.softmax(torch.relu(self.emb @ self.emb.T), dim=-1)

    def alpha(self):
        return torch.sigmoid(self.alpha_logit)

    def forward(self, physical_adj):
        a = self.alpha()
        return (1 - a) * physical_adj + a * self.learned()


class Profile(nn.Module):
    """Per-sensor time-of-day profile (train-only historical mean, z-space) as features.

    inputs()        -> profile value at every input step           [B, L, N]
    future_change() -> profile(t+k) - profile(t), k = 1..H          [B, N, H]
    This is the information that made the LightGBM model strong: *when* a given
    sensor usually slows down."""

    def __init__(self, profile, horizon):
        super().__init__()
        self.register_buffer('profile', torch.as_tensor(profile, dtype=torch.float32))
        self.horizon = horizon

    def _slots(self, time_feats):
        from .models_modern import calendar_index
        return calendar_index(time_feats, self.profile.shape[0])[0]          # [B,L]

    def inputs(self, time_feats):
        return self.profile[self._slots(time_feats)]                           # [B,L,N]

    def future_change(self, time_feats):
        s_last = self._slots(time_feats)[:, -1]                                # [B]
        k = torch.arange(1, self.horizon + 1, device=s_last.device)
        fut = self.profile[(s_last[:, None] + k[None, :]) % self.profile.shape[0]]   # [B,H,N]
        return (fut - self.profile[s_last].unsqueeze(1)).transpose(1, 2)       # [B,N,H]


def make_profile(bundle, horizon):
    from .data import tod_profile_z
    if getattr(bundle, 'scaled', None) is not None:
        prof = tod_profile_z(bundle)
    else:  # minimal bundle (predict.py): the buffer is restored from the checkpoint
        prof = np.zeros((int(round(1440 / bundle.interval_min)), len(bundle.sensor_ids)), np.float32)
    return Profile(prof, horizon)


def _append_time(xt, time_feats, step, n):
    t = time_feats[:, step].unsqueeze(1).expand(-1, n, -1)
    return torch.cat([xt, t], dim=-1)


class TGCN(nn.Module):
    def __init__(self, num_nodes, adj, hidden_dim=64, horizon=3,
                 time_features=False, adaptive_graph=False, residual=False,
                 dropout=0.0, graph_embed_dim=8, graph=True, self_sep=False, profile=None):
        super().__init__()
        self.prof = profile
        self.register_buffer('adj', adj.clone().detach() if graph
                             else torch.eye(num_nodes, dtype=adj.dtype))
        self.num_nodes = num_nodes
        self.horizon = horizon
        self.use_time = bool(time_features)
        self.use_adaptive = bool(adaptive_graph)
        self.use_residual = bool(residual)
        self.graph = AdaptiveGraph(num_nodes, graph_embed_dim) if self.use_adaptive else None
        self.cell = TGCNCell(1 + (4 if self.use_time else 0) + (1 if profile is not None else 0),
                             hidden_dim, self_sep=self_sep)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim + (horizon if profile is not None else 0), horizon)

    def used_adjacency(self):
        if self.use_adaptive:
            return self.graph(self.adj)
        cache = getattr(self, '_sparse_adj', None)
        if cache is None or cache.device != self.adj.device:
            cache = self.adj.to_sparse().coalesce()
            object.__setattr__(self, '_sparse_adj', cache)  # not saved in state_dict
        return cache

    def forward(self, x, time_feats=None, return_gates=False):
        b, l, n, c = x.shape
        assert n == self.num_nodes and c == 1
        if self.use_time and (time_feats is None or time_feats.shape != (b, l, 4)):
            raise ValueError('time_features=True requires time_feats with shape [B,L,4].')
        adj = self.used_adjacency()
        h = x.new_zeros(b, n, self.cell.hidden_dim)
        gates = []
        prof_in = self.prof.inputs(time_feats) if self.prof is not None else None
        for step in range(l):
            xt = x[:, step]
            if self.use_time:
                xt = _append_time(xt, time_feats, step, n)
            if prof_in is not None:
                xt = torch.cat([xt, prof_in[:, step].unsqueeze(-1)], dim=-1)
            if return_gates:
                h, g = self.cell(xt, h, adj, return_gates=True)
                gates.append(g)
            else:
                h = self.cell(xt, h, adj)
        z = self.dropout(h)
        if self.prof is not None:
            z = torch.cat([z, self.prof.future_change(time_feats)], dim=-1)
        pred = self.head(z).transpose(1, 2)  # [B,H,N]
        if self.use_residual:
            pred = pred + x[:, -1, :, 0].unsqueeze(1)
        if return_gates:
            return pred, gates, h
        return pred


class NodeGRU(nn.Module):
    """Shared temporal GRU applied to each sensor independently (no graph).

    Same as the GRU baseline of the authors: one set of weights for every
    sensor, no information exchange between sensors.
    """

    def __init__(self, num_nodes, adj, hidden_dim=64, horizon=3,
                 time_features=False, residual=False, dropout=0.0, profile=None, **kwargs):
        super().__init__()
        self.num_nodes = num_nodes
        self.horizon = horizon
        self.use_time = bool(time_features)
        self.use_residual = bool(residual)
        self.prof = profile
        extra = 1 if profile is not None else 0
        self.gru = nn.GRU(1 + (4 if self.use_time else 0) + extra, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim + (horizon if profile is not None else 0), horizon)

    def forward(self, x, time_feats=None):
        b, l, n, _ = x.shape
        seq = x
        if self.use_time:
            seq = torch.cat([x, time_feats.unsqueeze(2).expand(-1, -1, n, -1)], dim=-1)
        if self.prof is not None:
            seq = torch.cat([seq, self.prof.inputs(time_feats).unsqueeze(-1)], dim=-1)
        seq = seq.permute(0, 2, 1, 3).reshape(b * n, l, seq.shape[-1])
        _, h = self.gru(seq)
        z = self.dropout(h[-1]).reshape(b, n, -1)
        if self.prof is not None:
            z = torch.cat([z, self.prof.future_change(time_feats)], dim=-1)
        pred = self.head(z).transpose(1, 2)
        if self.use_residual:
            pred = pred + x[:, -1, :, 0].unsqueeze(1)
        return pred


class GCN(nn.Module):
    """Spatial-only baseline of the paper: the L input steps are treated as node
    features; two graph convolutions, no recurrence (no temporal state)."""

    def __init__(self, num_nodes, adj, hidden_dim=64, horizon=3, seq_len=12, **kwargs):
        super().__init__()
        self.register_buffer('adj', adj.clone().detach())
        self.horizon = horizon
        self.l1 = GraphLinear(seq_len, hidden_dim)
        self.l2 = GraphLinear(hidden_dim, horizon)

    def forward(self, x, time_feats=None):
        h = x[..., 0].transpose(1, 2)                   # [B,N,L]
        adj = self.adj.to_sparse()
        h = torch.relu(self.l1(h, adj))
        return self.l2(h, adj).transpose(1, 2)           # [B,H,N]


MODEL_NAMES = ('gru', 'gcn', 'tgcn', 'tgcn_plus', 'stid', 'gwnet', 'dcrnn', 'staeformer')


def make_model(model_name, bundle, hidden_dim=64, horizon=3,
               no_time_features=False, no_adaptive_graph=False,
               no_residual=False, dropout=0.1, graph_embed_dim=8,
               seq_len=12, time_features=False, residual=False,
               adaptive_graph=False, no_graph=False, self_sep=False, size='compact', profile=False):
    """'tgcn_plus' = T-GCN with time features + adaptive graph + residual (all on,
    switch off with no_*). 'gru'/'tgcn' are the plain paper models unless the
    explicit time_features/residual/adaptive_graph switches are given."""
    from .models_modern import MODERN, make_modern
    if model_name in MODERN:
        return make_modern(model_name, bundle, seq_len, horizon, hidden_dim=hidden_dim,
                           residual=residual, size=size)
    adj = torch.as_tensor(bundle.adj, dtype=torch.float32)
    n = len(bundle.sensor_ids)
    prof = make_profile(bundle, horizon) if profile else None
    if model_name == 'gru':
        return NodeGRU(n, adj, hidden_dim, horizon, time_features=time_features,
                       residual=residual, dropout=dropout if (time_features or residual) else 0.0,
                       profile=prof)
    if model_name == 'gcn':
        return GCN(n, adj, hidden_dim, horizon, seq_len=seq_len)
    if model_name == 'tgcn':
        return TGCN(n, adj, hidden_dim, horizon, time_features=time_features,
                    adaptive_graph=adaptive_graph, residual=residual,
                    dropout=dropout if (time_features or residual or adaptive_graph) else 0.0,
                    graph_embed_dim=graph_embed_dim, graph=not no_graph, self_sep=self_sep,
                    profile=prof)
    if model_name == 'tgcn_plus':
        return TGCN(n, adj, hidden_dim, horizon,
                    time_features=not no_time_features,
                    adaptive_graph=not no_adaptive_graph,
                    residual=not no_residual,
                    dropout=dropout, graph_embed_dim=graph_embed_dim, graph=not no_graph)
    raise ValueError(f'Unsupported model: {model_name}')
