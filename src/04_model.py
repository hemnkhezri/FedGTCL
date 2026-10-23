# Paper output: Section 3.3 (module: FedGTCL encoder)
# Run from the repository root.
"""
Step 4: Lightweight graph-temporal encoder (Section 3.3).

Implementation notes:
- The sparsified graph-attention layer (Eqs. 2-3) is implemented with DENSE
  masked attention (NxN attention matrix, masked to the top-m neighborhood
  computed in 03_graph_construction.py) rather than sparse scatter ops.
  For the small per-window graphs actually produced from real IIoT traffic
  (7-20 hosts per 60s window, verified empirically), this is numerically
  identical to a sparse implementation and avoids an external dependency
  (torch-geometric) whose compiled extensions are not reliably installable
  in this environment.
- The depthwise-separable temporal convolution (Eq. 4) is implemented with
  torch.nn.Conv1d(groups=channels) followed by a 1x1 pointwise Conv1d.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SparsifiedGraphAttention(nn.Module):
    """Eqs. (2)-(3): dense masked graph attention over a top-m neighborhood."""

    def __init__(self, in_dim, out_dim, low_rank_dim=None):
        super().__init__()
        r = low_rank_dim or out_dim
        # low-rank projection W = W2 @ W1  (Section 3.3a: "shared, low-rank
        # projection matrix", reducing parameters relative to a full-rank W)
        self.W1 = nn.Linear(in_dim, r, bias=False)
        self.W2 = nn.Linear(r, out_dim, bias=False)
        self.a = nn.Linear(2 * out_dim, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.out_dim = out_dim

    def project(self, h):
        return self.W2(self.W1(h))

    def forward(self, h, adj_mask):
        """
        h        : [N, in_dim]
        adj_mask : [N, N] bool, True where column node is in row node's
                   top-m neighborhood (self excluded)
        returns  : [N, out_dim]
        """
        N = h.size(0)
        Wh = self.project(h)                                  # [N, out_dim]
        Wh_i = Wh.unsqueeze(1).expand(N, N, self.out_dim)      # [N, N, out_dim]
        Wh_j = Wh.unsqueeze(0).expand(N, N, self.out_dim)      # [N, N, out_dim]
        e = self.leaky_relu(self.a(torch.cat([Wh_i, Wh_j], dim=-1))).squeeze(-1)  # [N, N]

        e = e.masked_fill(~adj_mask, float("-inf"))
        # rows with no retained neighbor (isolated node) -> self-loop fallback
        isolated = (~adj_mask).all(dim=1)
        if isolated.any():
            e[isolated, isolated] = 0.0
            adj_mask = adj_mask.clone()
            adj_mask[isolated, isolated] = True
            e = e.masked_fill(~adj_mask, float("-inf"))

        alpha = F.softmax(e, dim=1)                            # [N, N]
        out = torch.matmul(alpha, Wh)                          # [N, out_dim]
        return F.elu(out)


class AttentionReadout(nn.Module):
    """Graph-level readout: attention-weighted pooling against a learned query."""

    def __init__(self, dim):
        super().__init__()
        self.query = nn.Parameter(torch.randn(dim) * 0.1)
        self.key = nn.Linear(dim, dim, bias=False)

    def forward(self, h):
        # h: [N, dim] -> graph embedding [dim]
        scores = (self.key(h) @ self.query) / (h.size(-1) ** 0.5)   # [N]
        alpha = F.softmax(scores, dim=0)
        return (alpha.unsqueeze(-1) * h).sum(dim=0)                  # [dim]


class DepthwiseSeparableTemporalConv(nn.Module):
    """Eq. (4): depthwise conv over the temporal axis + 1x1 pointwise mix."""

    def __init__(self, channels, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        self.depthwise = nn.Conv1d(channels, channels, kernel_size,
                                    padding=pad, groups=channels, bias=False)
        self.pointwise = nn.Conv1d(channels, channels, 1, bias=True)

    def forward(self, x):
        # x: [1, channels, T] -> [1, channels, T]
        return self.pointwise(self.depthwise(x))


class ContrastiveProjectionHead(nn.Module):
    def __init__(self, dim, proj_dim=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, proj_dim))

    def forward(self, z):
        return F.normalize(self.net(z), dim=-1)


class ClassificationHead(nn.Module):
    def __init__(self, dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, z):
        return self.fc(z)  # logits; softmax applied in the loss


class FedGTCLEncoder(nn.Module):
    """Full pipeline of Fig. 5: GAT layer -> readout -> temporal conv -> heads."""

    def __init__(self, in_dim, hidden_dim=64, embed_dim=64, low_rank_dim=32,
                 num_classes=5, proj_dim=32, temporal_kernel=3):
        super().__init__()
        self.gat = SparsifiedGraphAttention(in_dim, hidden_dim, low_rank_dim)
        self.readout = AttentionReadout(hidden_dim)
        self.temporal = DepthwiseSeparableTemporalConv(hidden_dim, temporal_kernel)
        self.proj_head = ContrastiveProjectionHead(hidden_dim, proj_dim)
        self.cls_head = ClassificationHead(hidden_dim, num_classes)
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

    def encode_window(self, node_feats, adj_mask):
        """One window: node_feats [N,in_dim], adj_mask [N,N] -> graph embedding [hidden_dim]."""
        h = self.gat(node_feats, adj_mask)
        g = self.readout(h)
        return g

    def forward(self, window_seq_feats, window_seq_masks):
        """
        window_seq_feats : list of T tensors, each [N_t, in_dim]
        window_seq_masks  : list of T tensors, each [N_t, N_t]
        returns z (temporal embedding), contrastive projection, class logits
        """
        graph_embeds = [self.encode_window(f, m) for f, m in zip(window_seq_feats, window_seq_masks)]
        seq = torch.stack(graph_embeds, dim=0)          # [T, hidden_dim]
        seq = seq.t().unsqueeze(0)                       # [1, hidden_dim, T]
        z_seq = self.temporal(seq)                        # [1, hidden_dim, T]
        z = z_seq[:, :, -1].squeeze(0)                    # take last step -> [hidden_dim]
        return z, self.proj_head(z), self.cls_head(z)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
