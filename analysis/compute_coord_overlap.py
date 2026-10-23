# Paper output: Section 3.5 (Jaccard overlap of clients' top-s coordinates; needs shared_partition_42.pkl)
# Run from the repository root.
import pickle, random, importlib.util, itertools
import numpy as np
import torch

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

model_mod = load_module("model_mod", "src/04_model.py")

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

with open("data/processed/shared_partition_42.pkl", "rb") as f:
    P = pickle.load(f)

IN_DIM = P["IN_DIM"]
windows_data = P["windows_data"]

with open("src/06_federated_training.py") as f:
    src = f.read()
# The published overlap values (Section 3.5: IID 0.31, std 0.09; non-IID 0.26, std 0.24) were computed while
# 06_federated_training.py still set SEED = 123. Executing its header re-seeds the RNGs, so the same value is
# used here to reproduce them exactly.
src = src.replace("\nSEED = 42\n", "\nSEED = 123\n")
main_idx = src.find('if __name__ == "__main__":')
ns = {}
exec(compile(src[:main_idx], "06_federated_training.py", "exec"), ns)

flatten_params = ns['flatten_params']
local_train = ns['local_train']
topk_sparsify = ns['topk_sparsify']

def get_client_topk_indices(client_train, s=0.3):
    model_template = model_mod.FedGTCLEncoder(in_dim=IN_DIM, hidden_dim=64, embed_dim=64,
                                                low_rank_dim=32, num_classes=2, proj_dim=32, temporal_kernel=3)
    global_flat = flatten_params(model_template).clone()
    index_sets = []
    for client_seqs in client_train:
        if len(client_seqs) < 2:
            index_sets.append(set())
            continue
        n_pos = sum(sq["label"] for sq in client_seqs)
        n_neg = len(client_seqs) - n_pos
        cw = torch.tensor([len(client_seqs)/max(1,2*n_neg), len(client_seqs)/max(1,2*n_pos)])
        delta, _ = local_train(global_flat, client_seqs, windows_data, model_template,
                             local_epochs=3, p_label=0.5, class_weights=cw)
        delta_sparse = topk_sparsify(delta, s=s)
        idx = set(torch.nonzero(delta_sparse).flatten().tolist())
        index_sets.append(idx)
    return index_sets

def pairwise_jaccard(index_sets):
    sims = []
    for a, b in itertools.combinations(range(len(index_sets)), 2):
        A, B = index_sets[a], index_sets[b]
        if not A and not B:
            continue
        if len(A) == 0 or len(B) == 0:
            sims.append(0.0)
            continue
        inter = len(A & B)
        union = len(A | B)
        sims.append(inter/union)
    return sims

print("Computing top-s coordinate overlap: IID vs non-IID (WUSTL-IIoT-2021, seed=42, s=0.3)...\n")

iid_idx = get_client_topk_indices(P["iid_train"], s=0.3)
sims_iid = pairwise_jaccard(iid_idx)
print(f"IID: {len(sims_iid)} client pairs, mean Jaccard overlap = {np.mean(sims_iid):.4f} (std={np.std(sims_iid):.4f})")
print(f"  per-pair: {[f'{v:.3f}' for v in sims_iid]}")

noniid_idx = get_client_topk_indices(P["noniid_train"], s=0.3)
sims_noniid = pairwise_jaccard(noniid_idx)
print(f"\nnon-IID: {len(sims_noniid)} client pairs, mean Jaccard overlap = {np.mean(sims_noniid):.4f} (std={np.std(sims_noniid):.4f})")
print(f"  per-pair: {[f'{v:.3f}' for v in sims_noniid]}")

print(f"\nRatio (non-IID / IID mean overlap): {np.mean(sims_noniid)/max(1e-9,np.mean(sims_iid)):.3f}")
