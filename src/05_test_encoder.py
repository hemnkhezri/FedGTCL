# Paper output: Table 2 (parameter counts and forward-pass latency)
# Run from the repository root.
import sys, time, importlib.util
import numpy as np
import pandas as pd
import torch

spec = importlib.util.spec_from_file_location("graph_mod", "src/03_graph_construction.py")
graph_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(graph_mod)

spec2 = importlib.util.spec_from_file_location("model_mod", "src/04_model.py")
model_mod = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(model_mod)

# ---- 1) load real data and build T=5 consecutive real graph windows ----
df = pd.read_csv("data/processed/WUSTL-IIoT-2021_processed.csv")
reserved = {"StartTime", "LastTime", "SrcAddr", "DstAddr", "Sport", "Dport", "Target", "Traffic"}
feature_cols = [c for c in df.columns if c not in reserved]
IN_DIM = len(feature_cols)

df = graph_mod.make_time_windows(df, "StartTime", window_seconds=60)
window_ids = sorted(df["window_id"].unique())

T = 5          # temporal window length (Section 3.3c)
TOP_M = 8      # sparsified neighborhood size (Section 3.3a)
HIDDEN_DIM = 64
LOW_RANK = 32
NUM_CLASSES = df["Traffic"].nunique()

chosen_windows = window_ids[10:10 + T]  # arbitrary real slice, not cherry-picked for size
feats_seq, masks_seq = [], []
for w in chosen_windows:
    wdf = df[df["window_id"] == w]
    node_ids, node_feats, adj_mask, edge_weight = graph_mod.build_window_graph(
        wdf, "SrcAddr", "DstAddr", feature_cols, top_m=TOP_M
    )
    feats_seq.append(torch.tensor(node_feats, dtype=torch.float32))
    masks_seq.append(torch.tensor(adj_mask, dtype=torch.bool))
    print(f"window {w}: N={len(node_ids)} nodes")

# ---- 2) build the model and run a real forward pass ----
model = model_mod.FedGTCLEncoder(
    in_dim=IN_DIM, hidden_dim=HIDDEN_DIM, embed_dim=HIDDEN_DIM,
    low_rank_dim=LOW_RANK, num_classes=NUM_CLASSES, proj_dim=32, temporal_kernel=3,
)
model.eval()

n_params = model_mod.count_parameters(model)
print(f"\nModel parameter count (real, counted): {n_params}")

with torch.no_grad():
    z, proj, logits = model(feats_seq, masks_seq)
print("Output shapes -> z:", z.shape, " proj:", proj.shape, " logits:", logits.shape)

# ---- 3) measure real forward-pass latency (CPU, mean of 50 runs) ----
N_RUNS = 50
with torch.no_grad():
    # warmup
    for _ in range(5):
        model(feats_seq, masks_seq)
    t0 = time.perf_counter()
    for _ in range(N_RUNS):
        model(feats_seq, masks_seq)
    t1 = time.perf_counter()
latency_ms = 1000 * (t1 - t0) / N_RUNS
print(f"\nMeasured mean forward-pass latency over {N_RUNS} runs (CPU): {latency_ms:.3f} ms "
      f"per T={T}-window sequence (host counts: {[f.shape[0] for f in feats_seq]})")

# ---- 4) compare against a dense (non-sparsified, non-low-rank) GCN-GRU-style
#          baseline layer of comparable width, to quantify the reduction
#          argued in Section 3.6 ----
class DenseGCNGRUBaseline(torch.nn.Module):
    """Fair dense-GCN baseline: true full-adjacency aggregation (no top-m
    sparsification), full-rank weights (no low-rank factorization), followed
    by a real GRU -- mirroring GACO's GCN-GRU design choices (Section 3.3)
    so the parameter/latency comparison in Section 3.6 is apples-to-apples."""
    def __init__(self, in_dim, hidden_dim, num_classes):
        super().__init__()
        self.gcn1 = torch.nn.Linear(in_dim, hidden_dim)       # full-rank W (no low-rank factorization)
        self.gcn2 = torch.nn.Linear(hidden_dim, hidden_dim)   # full-rank W
        self.gru = torch.nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.cls = torch.nn.Linear(hidden_dim, num_classes)

    def dense_gcn_layer(self, h, lin):
        # true dense GCN aggregation over the FULL (unmasked) adjacency,
        # mean-normalized, i.e. h' = relu( (1/N) * A_full @ (h @ W) )
        N = h.size(0)
        Wh = lin(h)                                  # [N, hidden]
        A_full = torch.ones(N, N) / N                 # dense, fully-connected, normalized
        return torch.relu(A_full @ Wh)

    def forward(self, feats_seq):
        graph_embeds = []
        for f in feats_seq:
            h = self.dense_gcn_layer(f, self.gcn1)
            h = self.dense_gcn_layer(h, self.gcn2)
            graph_embeds.append(h.mean(dim=0))  # dense mean-pool readout
        seq = torch.stack(graph_embeds, dim=0).unsqueeze(0)  # [1, T, hidden]
        out, _ = self.gru(seq)
        return self.cls(out[:, -1, :])

baseline = DenseGCNGRUBaseline(IN_DIM, HIDDEN_DIM, NUM_CLASSES)
n_params_baseline = model_mod.count_parameters(baseline)
print(f"\nDense GCN-GRU-style baseline parameter count: {n_params_baseline}")
print(f"FedGTCL encoder parameter count:                {n_params}")
print(f"Reduction: {100*(1 - n_params/n_params_baseline):.1f}% fewer parameters")

with torch.no_grad():
    for _ in range(5):
        baseline(feats_seq)
    t0 = time.perf_counter()
    for _ in range(N_RUNS):
        baseline(feats_seq)
    t1 = time.perf_counter()
baseline_latency_ms = 1000 * (t1 - t0) / N_RUNS
print(f"Dense GCN-GRU-style baseline latency: {baseline_latency_ms:.3f} ms")
print(f"FedGTCL encoder latency:              {latency_ms:.3f} ms")
