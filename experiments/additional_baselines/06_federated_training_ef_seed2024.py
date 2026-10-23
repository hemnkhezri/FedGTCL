# Paper output: Online Resource 1, Table S5, seed 2024 (needs shared_partition_2024.pkl)
# Run from the repository root.
"""
Step 6: Full federated training pipeline for FedGTCL (Sections 3.4, 3.5,
Algorithm 1), run end-to-end on real WUSTL-IIoT-2021 graphs.

Scope of this run (explicitly, for the manuscript): binary (benign/attack)
window-level classification, K simulated clients, IID and non-IID
(Dirichlet) partitions. Multi-class and the other two datasets are
extended in the following iteration.
"""
import importlib.util
import random
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

graph_mod = load_module("graph_mod", "src/03_graph_construction.py")
model_mod = load_module("model_mod", "src/04_model.py")

SEED = 123
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ---------------------------------------------------------------------
# 1) Load the SHARED partition (identical client assignment AND identical
#    80/20 train/val split, reused by every method -- FedGTCL and both
#    FedAvg baselines -- for a fair, apples-to-apples comparison).
# ---------------------------------------------------------------------
import pickle
print("Loading shared partition (shared_partition_2024.pkl) ...")
with open("data/processed/shared_partition_2024.pkl", "rb") as f:
    _shared = pickle.load(f)

windows_data = _shared["windows_data"]
IN_DIM = _shared["IN_DIM"]
T = _shared["T"]
iid_train, iid_val = _shared["iid_train"], _shared["iid_val"]
noniid_train, noniid_val = _shared["noniid_train"], _shared["noniid_val"]
print(f"Loaded {len(windows_data)} real graphs, IN_DIM={IN_DIM}, T={T}")
print(f"IID   train sizes: {[len(t) for t in iid_train]}  val sizes: {[len(v) for v in iid_val]}")
print(f"non-IID train sizes: {[len(t) for t in noniid_train]}  val sizes: {[len(v) for v in noniid_val]}")

# 3) Contrastive augmentations (Section 3.4)
# ---------------------------------------------------------------------
def augment_edge_dropout(mask, rho=0.2):
    """Randomly drop a fraction rho of retained edges, guaranteeing each
    node keeps at least one neighbor (falls back to self-loop)."""
    mask = mask.clone()
    N = mask.size(0)
    drop = torch.rand_like(mask, dtype=torch.float32) < rho
    mask = mask & ~drop
    isolated = ~mask.any(dim=1)
    if isolated.any():
        mask[isolated, isolated] = True
    return mask

def augment_temporal_jitter(seq_dicts, all_windows, delta=1):
    """Shift the sequence start by up to +-delta positions if in range."""
    anchor = seq_dicts[-1]
    shift = random.randint(-delta, delta)
    start = all_windows.index(seq_dicts[0]) + shift
    start = max(0, min(start, len(all_windows) - T))
    return all_windows[start:start + T]

def build_view(seq_entry, all_windows, augment=True):
    seq = seq_entry["seq"]
    if augment:
        seq = augment_temporal_jitter(seq, all_windows, delta=1)
    feats = [w["feats"] for w in seq]
    masks = [augment_edge_dropout(w["mask"], rho=0.2) if augment else w["mask"] for w in seq]
    host_set = set(seq[-1]["node_ids"])
    return feats, masks, host_set

def jaccard(a, b):
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))

# ---------------------------------------------------------------------
# 4) Structure-aware contrastive loss (Eqs. 5, 5a) + supervised loss (Eq. 6)
# ---------------------------------------------------------------------
def structure_aware_nt_xent(z_list, host_sets, tau=0.5):
    """z_list: list of 2B projected+normalized embeddings [proj_dim]
    (view A and view B for B anchors, interleaved a0,b0,a1,b1,...).
    host_sets: matching list of host-id sets used for structural reweighting."""
    B2 = len(z_list)
    Z = torch.stack(z_list, dim=0)              # [2B, proj_dim]
    sim = Z @ Z.t()                                # [2B, 2B] cosine (already normalized)
    sim = sim / tau
    loss = 0.0
    for a in range(B2):
        pos = a + 1 if a % 2 == 0 else a - 1
        gamma_weights = torch.ones(B2)
        for j in range(B2):
            if j == a:
                continue
            gamma_weights[j] = 1.0 - jaccard(host_sets[a], host_sets[j])  # Eq. 5a
        logits = sim[a].clone()
        mask = torch.ones(B2, dtype=torch.bool); mask[a] = False
        # structure-aware re-weighting of negatives (Eq. 5a applied to Eq. 5)
        weighted_exp = torch.exp(logits[mask]) * gamma_weights[mask]
        denom = weighted_exp.sum() + torch.exp(logits[pos])
        loss += -(logits[pos] - torch.log(denom + 1e-8))
    return loss / B2

# ---------------------------------------------------------------------
# 5) FedAdaptOpt (Section 3.5, Algorithm 1)
# ---------------------------------------------------------------------
def flatten_params(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()])

def load_flat_params(model, flat):
    i = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat[i:i + n].view_as(p))
        i += n

def topk_sparsify(delta, s=0.3):
    k = max(1, int(s * delta.numel()))
    vals, idx = torch.topk(delta.abs(), k)
    out = torch.zeros_like(delta)
    out[idx] = delta[idx]
    return out

def local_train(global_flat, client_seqs, all_windows, model_template,
                 local_epochs=5, lr=1e-3, beta_max=1.0, p_label=0.5,
                 class_weights=None):
    model = copy.deepcopy(model_template)
    load_flat_params(model, global_flat)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    n_labeled = max(2, int(len(client_seqs) * p_label))
    labeled_anchor_idxs = set(random.sample([e["anchor_idx"] for e in client_seqs],
                                             min(n_labeled, len(client_seqs))))

    for epoch in range(local_epochs):
        beta = beta_max * (epoch + 1) / local_epochs
        random.shuffle(client_seqs)
        MAX_BATCH = 16  # cap batch size: contrastive loss is O(B^2); keeps runtime tractable
        batch = client_seqs[:MAX_BATCH]
        if len(batch) < 2:
            continue
        opt.zero_grad()
        proj_list, host_sets, logits_list, labels_list = [], [], [], []
        for i, entry in enumerate(batch):
            for view in range(2):
                feats, masks, hosts = build_view(entry, all_windows, augment=True)
                z, proj, logits = model(feats, masks)
                proj_list.append(proj)
                host_sets.append(hosts)
                if view == 0:
                    logits_list.append(logits)
                    labels_list.append(entry["label"])
        loss_con = structure_aware_nt_xent(proj_list, host_sets, tau=0.5)

        # supervised loss computed ONLY over the client's locally labeled
        # subset D_k^L (Section 3.1), using class-balanced cross-entropy to
        # counter the benign-dominant label distribution (Section 4.1)
        sup_indices = [i for i, e in enumerate(batch) if e["anchor_idx"] in labeled_anchor_idxs]
        loss_sup = torch.tensor(0.0)
        if len(sup_indices) >= 1:
            sup_logits = torch.stack([logits_list[i] for i in sup_indices], dim=0)
            sup_labels = torch.tensor([labels_list[i] for i in sup_indices], dtype=torch.long)
            loss_sup = F.cross_entropy(sup_logits, sup_labels, weight=class_weights)

        loss = loss_con + beta * loss_sup
        loss.backward()
        opt.step()

    local_flat = flatten_params(model)
    return local_flat - global_flat, model


def evaluate(model, flat_params, seqs, all_windows):
    load_flat_params(model, flat_params)
    model.eval()
    TP = FP = TN = FN = 0
    with torch.no_grad():
        for entry in seqs:
            feats, masks, _ = build_view(entry, all_windows, augment=False)
            _, _, logits = model(feats, masks)
            pred = int(logits.argmax())
            y = entry["label"]
            if pred == 1 and y == 1: TP += 1
            elif pred == 1 and y == 0: FP += 1
            elif pred == 0 and y == 0: TN += 1
            elif pred == 0 and y == 1: FN += 1
    acc = 100 * (TP + TN) / max(1, TP + TN + FP + FN)
    prec = 100 * TP / max(1, TP + FP)
    rec = 100 * TP / max(1, TP + FN)
    f1 = 2 * prec * rec / max(1e-8, prec + rec)
    fnr = 100 * FN / max(1, TP + FN)
    return dict(Accuracy=acc, Precision=prec, Recall=rec, F1=f1, FNR=fnr,
                TP=TP, FP=FP, TN=TN, FN=FN)


def run_federated(partition_name, client_train, client_val, all_windows,
                   R=15, local_epochs=5, s=0.3, eta_s=0.02, beta1=0.9, beta2=0.99,
                   p_label=0.5):
    """NOTE: client_train/client_val are now taken directly from the SHARED
    partition (data/processed/shared_partition_2024.pkl), identical to what both
    FedAvg baselines consume. This function no longer re-derives its own
    train/val split, which previously caused FedGTCL to be evaluated on a
    different validation-set composition than the baselines."""
    model_template = model_mod.FedGTCLEncoder(
        in_dim=IN_DIM, hidden_dim=64, embed_dim=64, low_rank_dim=32,
        num_classes=2, proj_dim=32, temporal_kernel=3,
    )
    global_flat = flatten_params(model_template).clone()
    m_t = torch.zeros_like(global_flat)
    v_t = torch.zeros_like(global_flat)
    eps = 1e-8

    # --- error-feedback: one residual buffer per client, carried across rounds ---
    residuals = [torch.zeros_like(global_flat) for _ in range(len(client_train))]

    # global class balance (for class-weighted supervised loss)
    all_train_labels = [s["label"] for tr in client_train for s in tr]
    n_pos = sum(all_train_labels); n_neg = len(all_train_labels) - n_pos
    class_weights = torch.tensor([len(all_train_labels) / max(1, 2 * n_neg),
                                    len(all_train_labels) / max(1, 2 * n_pos)])
    print(f"  class weights (benign, attack): {class_weights.tolist()}")

    print(f"\n--- Federated run: {partition_name} ---")
    for k, tr in enumerate(client_train):
        n_attack = sum(s["label"] for s in tr)
        print(f"  client {k}: train={len(tr)} (attack={n_attack}), val={len(client_val[k])}")

    for rnd in range(R):
        deltas, n_k_list = [], []
        for k in range(len(client_train)):
            if len(client_train[k]) < 2:
                continue
            delta, _ = local_train(global_flat, client_train[k], all_windows,
                                    model_template, local_epochs=local_epochs,
                                    p_label=p_label, class_weights=class_weights)
            # error-feedback: fold in this client's carried residual before sparsifying,
            # then keep whatever gets dropped this round for next time (Stich et al. 2018)
            delta_with_memory = delta + residuals[k]
            delta_sparse = topk_sparsify(delta_with_memory, s=s)
            residuals[k] = delta_with_memory - delta_sparse
            deltas.append(delta_sparse)
            n_k_list.append(len(client_train[k]))
        n_total = sum(n_k_list)
        agg_delta = sum((n_k / n_total) * d for n_k, d in zip(n_k_list, deltas))

        m_t = beta1 * m_t + (1 - beta1) * agg_delta
        v_t = beta2 * v_t + (1 - beta2) * agg_delta ** 2
        global_flat = global_flat + eta_s * m_t / (torch.sqrt(v_t) + eps)

        if (rnd + 1) % 3 == 0 or rnd == R - 1:
            all_val = [s for cv in client_val for s in cv]
            metrics = evaluate(model_template, global_flat, all_val, all_windows)
            print(f"  round {rnd+1}/{R}: val Accuracy={metrics['Accuracy']:.2f}% "
                  f"F1={metrics['F1']:.2f}% Recall={metrics['Recall']:.2f}% "
                  f"FNR={metrics['FNR']:.2f}%  (TP={metrics['TP']},FP={metrics['FP']},"
                  f"TN={metrics['TN']},FN={metrics['FN']})")

    all_val = [s for cv in client_val for s in cv]
    final_metrics = evaluate(model_template, global_flat, all_val, all_windows)
    return final_metrics


if __name__ == "__main__":
    results = {}
    results["IID"] = run_federated("IID", iid_train, iid_val, windows_data,
                                    R=10, local_epochs=3, s=0.3, eta_s=0.02, p_label=0.5)
    results["non-IID"] = run_federated("non-IID (Dirichlet a=0.3)", noniid_train, noniid_val,
                                        windows_data,
                                        R=10, local_epochs=3, s=0.3, eta_s=0.02, p_label=0.5)

    print("\n\n=== FINAL RESULTS (real, WUSTL-IIoT-2021, binary, K=4, R=10, SHARED partition) ===")
    for k, v in results.items():
        print(f"{k}: {v}")
