# Paper output: Section 3.6 (parameter accounting behind the 42.2% reduction)
# Run from the repository root.
import torch
import torch.nn as nn
import sys
sys.path.insert(0, 'src')
import importlib.util
spec = importlib.util.spec_from_file_location("model_mod", "src/04_model.py")
model_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_mod)

IN_DIM = 41
HIDDEN = 64
EMBED = 64
PROJ = 32
T = 5

def count_params(m):
    return sum(p.numel() for p in m.parameters())

# 1) Actual FedGTCL encoder (low-rank r=32, depthwise-separable temporal conv)
fedgtcl = model_mod.FedGTCLEncoder(in_dim=IN_DIM, hidden_dim=HIDDEN, embed_dim=EMBED,
                                     low_rank_dim=32, num_classes=2, proj_dim=PROJ, temporal_kernel=3)
n_fedgtcl = count_params(fedgtcl)
print(f"Full FedGTCL encoder (low-rank r=32, depthwise-conv temporal): {n_fedgtcl:,} params")

# 2) Variant: full-rank (r=64=hidden_dim, i.e. no low-rank compression), same depthwise-conv temporal
fedgtcl_fullrank = model_mod.FedGTCLEncoder(in_dim=IN_DIM, hidden_dim=HIDDEN, embed_dim=EMBED,
                                              low_rank_dim=HIDDEN, num_classes=2, proj_dim=PROJ, temporal_kernel=3)
n_fullrank = count_params(fedgtcl_fullrank)
print(f"Full-rank variant (r=64, depthwise-conv temporal):            {n_fullrank:,} params")
print(f"  -> low-rank projection saving (r=32 vs r=64): {n_fullrank - n_fedgtcl:,} params")

# 3) Isolate the GRU-vs-depthwise-conv saving directly: count each temporal module alone
gru_temporal = nn.GRU(HIDDEN, HIDDEN, batch_first=True)
n_gru = count_params(gru_temporal)
depthwise_temporal = model_mod.DepthwiseSeparableTemporal(HIDDEN, kernel_size=3) if hasattr(model_mod, 'DepthwiseSeparableTemporal') else None
print(f"\nStandalone GRU(64,64) temporal module:            {n_gru:,} params")
if depthwise_temporal is not None:
    n_dwc = count_params(depthwise_temporal)
    print(f"Standalone depthwise-separable temporal (T=5, kernel=3): {n_dwc:,} params")
    print(f"  -> temporal-architecture saving (GRU -> depthwise-conv): {n_gru - n_dwc:,} params")

# 4) Full dense baseline as actually used in this study's Table 2 comparison
#    (full-rank attention + GRU temporal) for the total reduction check
total_dense_estimate = n_fullrank - (n_dwc if depthwise_temporal is not None else 0) + n_gru
print(f"\nEstimated full dense baseline (full-rank attn + GRU temporal): {total_dense_estimate:,} params")
print(f"Measured dense baseline (Table 2): 32,133 params")
print(f"Measured FedGTCL encoder (Table 2): 18,565 params")
print(f"Total measured saving: {32133-18565:,} params")
