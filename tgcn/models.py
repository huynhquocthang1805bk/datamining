"""T-GCN: graph convolution inside GRU gates; extended model is experimental.

This code is an independent implementation motivated by Zhao et al. (2019),
not a byte-for-byte reproduction of the original TensorFlow code.
"""
import torch
from torch import nn


class GraphLinear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x, adj):
        # x: [B,N,C]; adjacency: [N,N]
        return self.linear(torch.matmul(adj, x))


class TGCNCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gates = GraphLinear(input_dim + hidden_dim, 2 * hidden_dim)
        self.candidate = GraphLinear(input_dim + hidden_dim, hidden_dim)

    def forward(self, x, h, adj):
        z, r = torch.sigmoid(self.gates(torch.cat((x, h), dim=-1), adj)).chunk(2, -1)
        h_hat = torch.tanh(self.candidate(torch.cat((x, r*h), dim=-1), adj))
        return z*h + (1-z)*h_hat


class AdaptiveGraph(nn.Module):
    """Learn soft adjacency from node embeddings; mixture with physical graph.

    Note: unrestricted adaptive graph is not the original T-GCN mechanism.
    """
    def __init__(self, num_nodes, embed_dim=8, alpha_init=-1.4):
        super().__init__()
        self.emb = nn.Parameter(0.1*torch.randn(num_nodes, embed_dim))
        self.alpha_logit = nn.Parameter(torch.tensor(alpha_init))

    def forward(self, physical_adj):
        affinities = torch.relu(self.emb @ self.emb.T)
        learned = torch.softmax(affinities, dim=-1)
        alpha = torch.sigmoid(self.alpha_logit)
        return (1-alpha)*physical_adj + alpha*learned


class TGCN(nn.Module):
    def __init__(self, num_nodes, adj, hidden_dim=32, horizon=3,
                 improved=False, time_features=True, adaptive_graph=True,
                 residual=True, dropout=0.1, graph_embed_dim=8):
        super().__init__()
        self.register_buffer('adj', adj.clone().detach())
        self.num_nodes = num_nodes
        self.horizon = horizon
        self.improved = improved
        self.use_time = bool(improved and time_features)
        self.use_adaptive = bool(improved and adaptive_graph)
        self.use_residual = bool(improved and residual)
        self.graph = AdaptiveGraph(num_nodes, graph_embed_dim) if self.use_adaptive else None
        self.cell = TGCNCell(1 + (4 if self.use_time else 0), hidden_dim)
        self.dropout = nn.Dropout(dropout if improved else 0.)
        self.head = nn.Linear(hidden_dim, horizon)

    def forward(self, x, time_feats=None):
        # x [batch, input_len, nodes, 1]; time_feats [batch,input_len,4]
        b, l, n, c = x.shape
        assert n == self.num_nodes and c == 1
        if self.use_time and (time_feats is None or time_feats.shape != (b,l,4)):
            raise ValueError('T-GCN+ requires time_feats with shape [B,L,4].')
        adj = self.graph(self.adj) if self.use_adaptive else self.adj
        h = x.new_zeros(b, n, self.cell.hidden_dim)
        for step in range(l):
            xt = x[:, step]
            if self.use_time:
                t = time_feats[:, step].unsqueeze(1).expand(-1,n,-1)
                xt = torch.cat([xt,t],dim=-1)
            h = self.cell(xt,h,adj)
        pred = self.head(self.dropout(h)).transpose(1,2)  # [B,H,N]
        if self.use_residual:
            pred = pred + x[:, -1, :, 0].unsqueeze(1)
        return pred


class NodeGRU(nn.Module):
    """Shared temporal GRU applied to each sensor independently (no graph)."""
    def __init__(self, num_nodes, adj, hidden_dim=32, horizon=3, **kwargs):
        super().__init__()
        self.num_nodes = num_nodes
        self.horizon = horizon
        self.gru = nn.GRU(1, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim,horizon)

    def forward(self, x, time_feats=None):
        b,l,n,_ = x.shape
        seq = x.permute(0,2,1,3).reshape(b*n,l,1)
        _, h = self.gru(seq)
        return self.head(h[-1]).reshape(b,n,self.horizon).transpose(1,2)


def make_model(model_name, bundle, hidden_dim=32, horizon=3,
               no_time_features=False, no_adaptive_graph=False,
               no_residual=False, dropout=0.1, graph_embed_dim=8):
    adj = torch.as_tensor(bundle.adj,dtype=torch.float32)
    if model_name == 'gru':
        return NodeGRU(len(bundle.sensor_ids), adj, hidden_dim, horizon)
    if model_name in ('tgcn','tgcn_plus'):
        return TGCN(len(bundle.sensor_ids), adj, hidden_dim, horizon,
                    improved=(model_name=='tgcn_plus'),
                    time_features=not no_time_features,
                    adaptive_graph=not no_adaptive_graph,
                    residual=not no_residual,
                    dropout=dropout, graph_embed_dim=graph_embed_dim)
    raise ValueError(f'Unsupported model: {model_name}')
