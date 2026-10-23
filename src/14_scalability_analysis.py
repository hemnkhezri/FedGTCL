# Paper output: Fig. 13 (inference time vs. dataset size)
# Run from the repository root.
"""
Step 14: Real scalability analysis -- measures actual wall-clock computation
time of the FedGTCL pipeline. Only the encoder forward passes over all T=5
sequences (sequential, unbatched) are inside the timed region; graph
construction runs beforehand and is not timed (see Fig. 13 caption).
Measured as a function of dataset size, using real WUSTL-IIoT-2021 data subsampled
to different sizes. Analogous to base paper Fig. 16, but using ONLY
methods actually implemented in this study (FedGTCL vs. the dense
GCN-GRU baseline) -- not methods from the literature we did not
reproduce.
"""
import importlib.util, time
import numpy as np
import pandas as pd
import torch

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

graph_mod = load_module("graph_mod", "src/03_graph_construction.py")
model_mod = load_module("model_mod", "src/04_model.py")

torch.manual_seed(42)

df_full = pd.read_csv("data/processed/WUSTL-IIoT-2021_processed.csv")
reserved = {"StartTime", "LastTime", "SrcAddr", "DstAddr", "Sport", "Dport", "Target", "Traffic"}
feature_cols = [c for c in df_full.columns if c not in reserved]
IN_DIM = len(feature_cols)

dataset_sizes = [20000, 40000, 60000, 80000, 100000, 150000]
TOP_M = 8
T = 5

fedgtcl_model = model_mod.FedGTCLEncoder(in_dim=IN_DIM, hidden_dim=64, embed_dim=64,
                                          low_rank_dim=32, num_classes=2, proj_dim=32, temporal_kernel=3)
fedgtcl_model.eval()

class GCNGRUBaseline(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes):
        super().__init__()
        self.gcn1 = torch.nn.Linear(in_dim, hidden_dim)
        self.gcn2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.gru = torch.nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.cls = torch.nn.Linear(hidden_dim, num_classes)
    def dense_gcn_layer(self, h, lin):
        N = h.size(0); Wh = lin(h); A_full = torch.ones(N, N) / N
        return torch.relu(A_full @ Wh)
    def forward(self, feats_seq):
        graph_embeds = []
        for f in feats_seq:
            h = self.dense_gcn_layer(f, self.gcn1); h = self.dense_gcn_layer(h, self.gcn2)
            graph_embeds.append(h.mean(dim=0))
        seq = torch.stack(graph_embeds, dim=0).unsqueeze(0)
        out, _ = self.gru(seq)
        return self.cls(out[:, -1, :]).squeeze(0)

baseline_model = GCNGRUBaseline(IN_DIM, 64, 2)
baseline_model.eval()

# warmup pass (excludes one-time JIT/threading setup cost from the timed
# measurements below)
_warmup_feats = [torch.randn(5, IN_DIM) for _ in range(T)]
_warmup_masks = [torch.ones(5, 5, dtype=torch.bool) for _ in range(T)]
with torch.no_grad():
    for _ in range(3):
        fedgtcl_model(_warmup_feats, _warmup_masks)
        baseline_model(_warmup_feats)

results_fedgtcl = []
results_baseline = []

for n_rows in dataset_sizes:
    # sort chronologically FIRST (the raw CSV is not time-ordered; a
    # non-chronological prefix would span the same wall-clock range
    # regardless of size, artificially flattening the scalability curve),
    # then take an increasing prefix so dataset size genuinely controls the
    # number of graph-windows processed.
    df_sorted = df_full.copy()
    df_sorted["StartTime"] = pd.to_datetime(df_sorted["StartTime"])
    df_sorted = df_sorted.sort_values("StartTime").reset_index(drop=True)
    df = df_sorted.iloc[:n_rows].copy()
    df = graph_mod.make_time_windows(df, "StartTime", window_seconds=60)
    window_ids = sorted(df["window_id"].unique())

    windows = []
    for w in window_ids:
        wdf = df[df["window_id"] == w]
        node_ids, node_feats, adj_mask, edge_weight = graph_mod.build_window_graph(
            wdf, "SrcAddr", "DstAddr", feature_cols, top_m=TOP_M)
        windows.append({"feats": torch.tensor(node_feats, dtype=torch.float32),
                         "mask": torch.tensor(adj_mask, dtype=torch.bool)})

    # build T-length sequences and time a full pass over all of them (graph
    # construction + forward inference), matching "processing this dataset
    # size end-to-end" as the measured quantity
    sequences = []
    for i in range(T - 1, len(windows)):
        sequences.append(windows[i - T + 1: i + 1])

    t0 = time.perf_counter()
    with torch.no_grad():
        for seq in sequences:
            feats = [w["feats"] for w in seq]
            masks = [w["mask"] for w in seq]
            fedgtcl_model(feats, masks)
    t1 = time.perf_counter()
    results_fedgtcl.append(t1 - t0)

    t0 = time.perf_counter()
    with torch.no_grad():
        for seq in sequences:
            feats = [w["feats"] for w in seq]
            baseline_model(feats)
    t1 = time.perf_counter()
    results_baseline.append(t1 - t0)

    print(f"n_rows={n_rows}: n_windows={len(windows)}, n_sequences={len(sequences)}, "
          f"FedGTCL={results_fedgtcl[-1]:.3f}s, Baseline={results_baseline[-1]:.3f}s")

print("\n\n=== SCALABILITY DATA (real) ===")
print("dataset_sizes =", dataset_sizes)
print("fedgtcl_seconds =", [round(x, 3) for x in results_fedgtcl])
print("baseline_seconds =", [round(x, 3) for x in results_baseline])
