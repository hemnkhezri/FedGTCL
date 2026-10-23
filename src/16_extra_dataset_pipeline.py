# Paper output: Online Resource 1, Table S7, FedGTCL rows; usage: <xiiotid|cicapt>
# Run from the repository root.
"""
Step 16: the unchanged FedGTCL pipeline of 09_edgeiiot_full_pipeline.py applied to the window graphs of
X-IIoTID or CICAPT-IIoT2024 (built by 16a_prepare_xiiotid.py / 16a_prepare_cicapt.py). Only the data loading
differs: windows are read from data/processed/<dataset>_windows.pkl instead of being built here. Sequences
(T=5), IID and Dirichlet (alpha=0.3) partitions, the 80/20 split, FedGTCL and FedAdaptOpt (K=4, R=10, E=3,
p_L=0.5, seed 42) are those of Sections 4.3-4.4. Saves data/processed/shared_partition_<dataset>.pkl.
"""
import importlib.util, random, copy, pickle, re
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

model_mod = load_module("model_mod", "src/04_model.py")

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

import sys
TOP_M = 8
MAX_NODES = 30
T = 5
DS = sys.argv[1]            # "xiiotid" or "cicapt"
print(f"Loading pre-built window graphs for {DS} ...")
with open(f"data/processed/{DS}_windows.pkl", "rb") as f:
    _W = pickle.load(f)
windows_data = []
for w in _W["windows"]:
    if w is None:
        continue
    windows_data.append({"id": w["win"],
                         "feats": torch.tensor(w["feats"], dtype=torch.float32),
                         "mask": torch.tensor(w["mask"], dtype=torch.bool),
                         "label": int(w["label"])})
IN_DIM = windows_data[0]["feats"].shape[1]
print(f"Loaded {len(windows_data)} window graphs (host-capped at {MAX_NODES}), feature dim {IN_DIM}. "
      f"Label balance: {sum(w['label'] for w in windows_data)} attack / {len(windows_data)} total")

def make_sequences(win_list, T):
    seqs = []
    for i in range(T - 1, len(win_list)):
        seq = win_list[i - T + 1: i + 1]
        seqs.append({"seq": seq, "label": win_list[i]["label"], "anchor_idx": i})
    return seqs

all_sequences = make_sequences(windows_data, T)
n_attack = sum(s["label"] for s in all_sequences)
print(f"Total T={T} sequences: {len(all_sequences)} (attack={n_attack}, benign={len(all_sequences)-n_attack})")

K = 4

def partition_iid(sequences, k):
    idx = list(range(len(sequences))); random.shuffle(idx)
    return [idx[i::k] for i in range(k)]

def partition_noniid_dirichlet(sequences, k, alpha=0.3, min_per_client=15):
    labels = np.array([s["label"] for s in sequences])
    idx_by_class = {c: np.where(labels == c)[0].tolist() for c in np.unique(labels)}
    for attempt in range(200):
        client_idx = [[] for _ in range(k)]
        for c, idxs in idx_by_class.items():
            shuffled = idxs.copy(); random.shuffle(shuffled)
            proportions = np.random.dirichlet(alpha=[alpha] * k)
            splits = (np.cumsum(proportions) * len(shuffled)).astype(int)[:-1]
            parts = np.split(shuffled, splits)
            for ci, part in enumerate(parts):
                client_idx[ci].extend(part.tolist())
        sizes = [len(c) for c in client_idx]
        if min(sizes) >= min_per_client:
            print(f"  Dirichlet partition accepted after {attempt+1} attempt(s), sizes={sizes}")
            return client_idx
    return client_idx

def make_train_val(client_idx_lists, sequences):
    train, val = [], []
    for idxs in client_idx_lists:
        idxs2 = idxs.copy(); random.shuffle(idxs2)
        cut = max(1, int(0.8 * len(idxs2)))
        train.append([sequences[i] for i in idxs2[:cut]])
        val.append([sequences[i] for i in idxs2[cut:]])
    return train, val

iid_clients = partition_iid(all_sequences, K)
noniid_clients = partition_noniid_dirichlet(all_sequences, K, alpha=0.3)
print("IID sizes:", [len(c) for c in iid_clients])
print("non-IID sizes:", [len(c) for c in noniid_clients])

iid_train, iid_val = make_train_val(iid_clients, all_sequences)
noniid_train, noniid_val = make_train_val(noniid_clients, all_sequences)

with open(f"data/processed/shared_partition_{DS}.pkl", "wb") as f:
    pickle.dump({"windows_data": windows_data, "all_sequences": all_sequences, "IN_DIM": IN_DIM, "T": T,
                 "iid_train": iid_train, "iid_val": iid_val,
                 "noniid_train": noniid_train, "noniid_val": noniid_val}, f)
print(f"Saved data/processed/shared_partition_{DS}.pkl")

def augment_edge_dropout(mask, rho=0.2):
    mask = mask.clone()
    drop = torch.rand_like(mask, dtype=torch.float32) < rho
    mask = mask & ~drop
    isolated = ~mask.any(dim=1)
    if isolated.any():
        mask[isolated, isolated] = True
    return mask

def build_view(seq_entry, augment=True):
    seq = seq_entry["seq"]
    feats = [w["feats"] for w in seq]
    masks = [augment_edge_dropout(w["mask"], rho=0.2) if augment else w["mask"] for w in seq]
    return feats, masks

def structure_aware_nt_xent(z_list, tau=0.5):
    B2 = len(z_list)
    Z = torch.stack(z_list, dim=0)
    sim = (Z @ Z.t()) / tau
    loss = 0.0
    for a in range(B2):
        pos = a + 1 if a % 2 == 0 else a - 1
        mask = torch.ones(B2, dtype=torch.bool); mask[a] = False
        weighted_exp = torch.exp(sim[a][mask])
        denom = weighted_exp.sum() + torch.exp(sim[a][pos])
        loss += -(sim[a][pos] - torch.log(denom + 1e-8))
    return loss / B2

def flatten_params(model): return torch.cat([p.data.view(-1) for p in model.parameters()])
def load_flat_params(model, flat):
    i = 0
    for p in model.parameters():
        n = p.numel(); p.data.copy_(flat[i:i + n].view_as(p)); i += n

def topk_sparsify(delta, s=0.3):
    k = max(1, int(s * delta.numel()))
    vals, idx = torch.topk(delta.abs(), k)
    out = torch.zeros_like(delta); out[idx] = delta[idx]
    return out

def local_train_fedgtcl(global_flat, client_seqs, model_template, local_epochs=3, lr=1e-3,
                         beta_max=1.0, p_label=0.5, class_weights=None):
    model = copy.deepcopy(model_template)
    load_flat_params(model, global_flat)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    n_labeled = max(2, int(len(client_seqs) * p_label))
    labeled_anchor_idxs = set(random.sample([e["anchor_idx"] for e in client_seqs],
                                             min(n_labeled, len(client_seqs))))
    MAX_BATCH = 16
    for epoch in range(local_epochs):
        beta = beta_max * (epoch + 1) / local_epochs
        random.shuffle(client_seqs)
        batch = client_seqs[:MAX_BATCH]
        if len(batch) < 2: continue
        opt.zero_grad()
        proj_list, logits_list, labels_list = [], [], []
        for entry in batch:
            for view in range(2):
                feats, masks = build_view(entry, augment=True)
                z, proj, logits = model(feats, masks)
                proj_list.append(proj)
                if view == 0:
                    logits_list.append(logits); labels_list.append(entry["label"])
        loss_con = structure_aware_nt_xent(proj_list, tau=0.5)
        sup_indices = [i for i, e in enumerate(batch) if e["anchor_idx"] in labeled_anchor_idxs]
        loss_sup = torch.tensor(0.0)
        if len(sup_indices) >= 1:
            sup_logits = torch.stack([logits_list[i] for i in sup_indices], dim=0)
            sup_labels = torch.tensor([labels_list[i] for i in sup_indices], dtype=torch.long)
            loss_sup = F.cross_entropy(sup_logits, sup_labels, weight=class_weights)
        loss = loss_con + beta * loss_sup
        loss.backward(); opt.step()
    return flatten_params(model) - global_flat

def evaluate_fedgtcl(model, flat_params, seqs):
    load_flat_params(model, flat_params)
    model.eval()
    TP = FP = TN = FN = 0
    with torch.no_grad():
        for entry in seqs:
            feats, masks = build_view(entry, augment=False)
            _, _, logits = model(feats, masks)
            pred = int(logits.argmax()); y = entry["label"]
            if pred == 1 and y == 1: TP += 1
            elif pred == 1 and y == 0: FP += 1
            elif pred == 0 and y == 0: TN += 1
            elif pred == 0 and y == 1: FN += 1
    acc = 100 * (TP + TN) / max(1, TP + TN + FP + FN)
    prec = 100 * TP / max(1, TP + FP)
    rec = 100 * TP / max(1, TP + FN)
    f1 = 2 * prec * rec / max(1e-8, prec + rec)
    fnr = 100 * FN / max(1, TP + FN)
    return dict(Accuracy=acc, Precision=prec, Recall=rec, F1=f1, FNR=fnr, TP=TP, FP=FP, TN=TN, FN=FN)

def run_fedgtcl(name, client_train, client_val, R=10, local_epochs=3, s=0.3, eta_s=0.02,
                 beta1=0.9, beta2=0.99, p_label=0.5):
    model_template = model_mod.FedGTCLEncoder(in_dim=IN_DIM, hidden_dim=64, embed_dim=64,
                                                low_rank_dim=32, num_classes=2, proj_dim=32,
                                                temporal_kernel=3)
    global_flat = flatten_params(model_template).clone()
    m_t = torch.zeros_like(global_flat); v_t = torch.zeros_like(global_flat); eps = 1e-8
    all_train_labels = [s2["label"] for tr in client_train for s2 in tr]
    n_pos = sum(all_train_labels); n_neg = len(all_train_labels) - n_pos
    class_weights = torch.tensor([len(all_train_labels) / max(1, 2 * n_neg),
                                    len(all_train_labels) / max(1, 2 * n_pos)])
    print(f"\n--- FedGTCL: {name} ---")
    for rnd in range(R):
        deltas, n_k_list = [], []
        for k in range(len(client_train)):
            if len(client_train[k]) < 2: continue
            delta = local_train_fedgtcl(global_flat, client_train[k], model_template,
                                         local_epochs=local_epochs, p_label=p_label,
                                         class_weights=class_weights)
            deltas.append(topk_sparsify(delta, s=s)); n_k_list.append(len(client_train[k]))
        n_total = sum(n_k_list)
        agg_delta = sum((n_k / n_total) * d for n_k, d in zip(n_k_list, deltas))
        m_t = beta1 * m_t + (1 - beta1) * agg_delta
        v_t = beta2 * v_t + (1 - beta2) * agg_delta ** 2
        global_flat = global_flat + eta_s * m_t / (torch.sqrt(v_t) + eps)
        if (rnd + 1) % 3 == 0 or rnd == R - 1:
            all_val = [s2 for cv in client_val for s2 in cv]
            m = evaluate_fedgtcl(model_template, global_flat, all_val)
            print(f"  round {rnd+1}/{R}: Accuracy={m['Accuracy']:.2f}% F1={m['F1']:.2f}% "
                  f"Recall={m['Recall']:.2f}% FNR={m['FNR']:.2f}%")
    all_val = [s2 for cv in client_val for s2 in cv]
    return evaluate_fedgtcl(model_template, global_flat, all_val)

if __name__ == "__main__":
    results = {}
    results["FedGTCL-IID"] = run_fedgtcl("IID", iid_train, iid_val, R=10, local_epochs=3)
    results["FedGTCL-nonIID"] = run_fedgtcl("non-IID", noniid_train, noniid_val, R=10, local_epochs=3)
    print(f"\n\n=== {DS} FedGTCL FINAL RESULTS ===")
    for k, v in results.items():
        print(f"{k}: {v}")
