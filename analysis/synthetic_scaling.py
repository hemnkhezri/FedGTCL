# Paper output: Online Resource 1, Table S4 (synthetic host-count scaling)
# Run from the repository root.
import time
import torch
import torch.nn as nn
import importlib.util

spec = importlib.util.spec_from_file_location("model_mod", "src/04_model.py")
model_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_mod)

torch.manual_seed(42)
IN_DIM, HIDDEN = 41, 64
M = 8  # sparsified neighborhood size, matches Table 4

def make_topm_mask(N, m):
    """Random adjacency, then keep only the top-m highest-weight neighbors per row."""
    scores = torch.rand(N, N)
    scores.fill_diagonal_(-1)  # exclude self before top-m
    k = min(m, N - 1) if N > 1 else 0
    mask = torch.zeros(N, N, dtype=torch.bool)
    if k > 0:
        topk_idx = scores.topk(k, dim=1).indices
        mask.scatter_(1, topk_idx, True)
    return mask

gat = model_mod.SparsifiedGraphAttention(IN_DIM, HIDDEN, low_rank_dim=32)
gat.eval()

dense_gcn = nn.Linear(IN_DIM, HIDDEN)
dense_gcn.eval()

V_values = [10, 20, 50, 100, 200, 400, 800]
n_repeats = 20

print(f"{'|V|':>6} | {'GAT (m=8) ms':>14} | {'Dense-GCN ms':>14} | {'GAT/Dense ratio':>16}")
print("-" * 60)
results = []
for N in V_values:
    h = torch.randn(N, IN_DIM)
    mask = make_topm_mask(N, M)

    # warm-up
    with torch.no_grad():
        _ = gat(h, mask)
        _ = dense_gcn(h)

    with torch.no_grad():
        t0 = time.perf_counter()
        for _ in range(n_repeats):
            _ = gat(h, mask)
        t_gat = (time.perf_counter() - t0) / n_repeats * 1000

        t0 = time.perf_counter()
        for _ in range(n_repeats):
            _ = dense_gcn(h)
        t_dense = (time.perf_counter() - t0) / n_repeats * 1000

    ratio = t_gat / t_dense if t_dense > 0 else float('nan')
    results.append((N, t_gat, t_dense, ratio))
    print(f"{N:>6} | {t_gat:>14.4f} | {t_dense:>14.4f} | {ratio:>16.2f}")

print("\nScaling check: does GAT time grow faster than dense-GCN time as |V| increases?")
print(f"GAT time |V|=800 / |V|=10 ratio: {results[-1][1]/results[0][1]:.1f}x  (|V| grew {800/10:.0f}x)")
print(f"Dense-GCN time |V|=800 / |V|=10 ratio: {results[-1][2]/results[0][2]:.1f}x  (|V| grew {800/10:.0f}x)")
