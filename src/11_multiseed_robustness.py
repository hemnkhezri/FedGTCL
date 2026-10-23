# Paper output: Table 12 and Fig. 10 (needs shared_partition_{42,123,2024}.pkl from experiments/sensitivity/07_shared_setup_param.py)
# Run from the repository root.
"""
Step 11: Multi-seed robustness check (Section 6.4 item 3) -- repeats the
core WUSTL-IIoT-2021 comparison (FedGTCL vs FedAvg-GCN-GRU) across 3
random seeds, at the identical pilot scale (K=4, R=10, E=3), to report
mean +/- std instead of a single point estimate, directly addressing the
single-seed limitation flagged in Section 6.4.
"""
import importlib.util, random, copy, pickle
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

SEEDS = [42, 123, 2024]

# ---------------------------------------------------------------------
# Rebuild the WUSTL windows/sequences ONCE (deterministic, seed-independent
# feature extraction); only the partition and training use the seed.
# ---------------------------------------------------------------------
df = pd.read_csv("data/processed/WUSTL-IIoT-2021_processed.csv")
reserved = {"StartTime", "LastTime", "SrcAddr", "DstAddr", "Sport", "Dport", "Target", "Traffic"}
feature_cols = [c for c in df.columns if c not in reserved]
IN_DIM = len(feature_cols)
df = graph_mod.make_time_windows(df, "StartTime", window_seconds=60)
window_ids = sorted(df["window_id"].unique())

TOP_M = 8
windows_data = []
for w in window_ids:
    wdf = df[df["window_id"] == w]
    node_ids, node_feats, adj_mask, edge_weight = graph_mod.build_window_graph(
        wdf, "SrcAddr", "DstAddr", feature_cols, top_m=TOP_M)
    label = int(wdf["Target"].max())
    windows_data.append({"node_ids": node_ids,
                          "feats": torch.tensor(node_feats, dtype=torch.float32),
                          "mask": torch.tensor(adj_mask, dtype=torch.bool), "label": label})

T = 5
def make_sequences(win_list):
    seqs = []
    for i in range(T - 1, len(win_list)):
        seq = win_list[i - T + 1: i + 1]
        seqs.append({"seq": seq, "label": win_list[i]["label"], "anchor_idx": i})
    return seqs
all_sequences = make_sequences(windows_data)
K = 4

# ---------------------------------------------------------------------
# Seed-dependent partitioning
# ---------------------------------------------------------------------
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

# ---------------------------------------------------------------------
# Contrastive + FedAdaptOpt (FedGTCL) -- same logic as 06_federated_training.py
# ---------------------------------------------------------------------
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

def flatten_params(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()])

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
    acc = 100*(TP+TN)/max(1,TP+TN+FP+FN); prec = 100*TP/max(1,TP+FP)
    rec = 100*TP/max(1,TP+FN); f1 = 2*prec*rec/max(1e-8,prec+rec)
    return dict(Accuracy=acc, Precision=prec, Recall=rec, F1=f1, TP=TP,FP=FP,TN=TN,FN=FN)

def run_fedgtcl(client_train, client_val, R=10, local_epochs=3, s=0.3, eta_s=0.02,
                beta1=0.9, beta2=0.99, p_label=0.5):
    model_template = model_mod.FedGTCLEncoder(in_dim=IN_DIM, hidden_dim=64, embed_dim=64,
                                                low_rank_dim=32, num_classes=2, proj_dim=32,
                                                temporal_kernel=3)
    global_flat = flatten_params(model_template).clone()
    m_t = torch.zeros_like(global_flat); v_t = torch.zeros_like(global_flat); eps = 1e-8
    all_train_labels = [s2["label"] for tr in client_train for s2 in tr]
    n_pos = sum(all_train_labels); n_neg = len(all_train_labels)-n_pos
    class_weights = torch.tensor([len(all_train_labels)/max(1,2*n_neg), len(all_train_labels)/max(1,2*n_pos)])
    for rnd in range(R):
        deltas, n_k_list = [], []
        for k in range(len(client_train)):
            if len(client_train[k]) < 2: continue
            delta = local_train_fedgtcl(global_flat, client_train[k], model_template,
                                         local_epochs=local_epochs, p_label=p_label, class_weights=class_weights)
            deltas.append(topk_sparsify(delta, s=s)); n_k_list.append(len(client_train[k]))
        n_total = sum(n_k_list)
        agg_delta = sum((n_k/n_total)*d for n_k, d in zip(n_k_list, deltas))
        m_t = beta1*m_t + (1-beta1)*agg_delta
        v_t = beta2*v_t + (1-beta2)*agg_delta**2
        global_flat = global_flat + eta_s*m_t/(torch.sqrt(v_t)+eps)
    all_val = [s2 for cv in client_val for s2 in cv]
    return evaluate_fedgtcl(model_template, global_flat, all_val)

# ---------------------------------------------------------------------
# FedAvg-GCN-GRU baseline
# ---------------------------------------------------------------------
class GCNGRUBaseline(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes):
        super().__init__()
        self.gcn1 = torch.nn.Linear(in_dim, hidden_dim)
        self.gcn2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.gru = torch.nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.cls = torch.nn.Linear(hidden_dim, num_classes)
    def dense_gcn_layer(self, h, lin):
        N = h.size(0); Wh = lin(h); A_full = torch.ones(N, N)/N
        return torch.relu(A_full @ Wh)
    def forward(self, feats_seq):
        graph_embeds = []
        for f in feats_seq:
            h = self.dense_gcn_layer(f, self.gcn1); h = self.dense_gcn_layer(h, self.gcn2)
            graph_embeds.append(h.mean(dim=0))
        seq = torch.stack(graph_embeds, dim=0).unsqueeze(0)
        out, _ = self.gru(seq)
        return self.cls(out[:, -1, :]).squeeze(0)

def local_train_fedavg(global_flat, client_seqs, model_template, local_epochs=3, lr=1e-3, class_weights=None):
    model = copy.deepcopy(model_template)
    load_flat_params(model, global_flat)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    MAX_BATCH = 16
    for epoch in range(local_epochs):
        random.shuffle(client_seqs)
        batch = client_seqs[:MAX_BATCH]
        if len(batch) < 1: continue
        opt.zero_grad()
        logits_list, labels_list = [], []
        for entry in batch:
            feats = [w["feats"] for w in entry["seq"]]
            logits_list.append(model(feats)); labels_list.append(entry["label"])
        logits_batch = torch.stack(logits_list, dim=0)
        labels_batch = torch.tensor(labels_list, dtype=torch.long)
        loss = F.cross_entropy(logits_batch, labels_batch, weight=class_weights)
        loss.backward(); opt.step()
    return flatten_params(model) - global_flat

def evaluate_fedavg(model, flat_params, seqs):
    load_flat_params(model, flat_params)
    model.eval()
    TP=FP=TN=FN=0
    with torch.no_grad():
        for entry in seqs:
            feats = [w["feats"] for w in entry["seq"]]
            logits = model(feats)
            pred = int(logits.argmax()); y = entry["label"]
            if pred==1 and y==1: TP+=1
            elif pred==1 and y==0: FP+=1
            elif pred==0 and y==0: TN+=1
            elif pred==0 and y==1: FN+=1
    acc=100*(TP+TN)/max(1,TP+TN+FP+FN); prec=100*TP/max(1,TP+FP)
    rec=100*TP/max(1,TP+FN); f1=2*prec*rec/max(1e-8,prec+rec)
    return dict(Accuracy=acc, Precision=prec, Recall=rec, F1=f1, TP=TP,FP=FP,TN=TN,FN=FN)

def run_fedavg_gcngru(client_train, client_val, R=10, local_epochs=3):
    model_template = GCNGRUBaseline(IN_DIM, 64, 2)
    global_flat = flatten_params(model_template).clone()
    all_train_labels = [s["label"] for tr in client_train for s in tr]
    n_pos = sum(all_train_labels); n_neg = len(all_train_labels)-n_pos
    class_weights = torch.tensor([len(all_train_labels)/max(1,2*n_neg), len(all_train_labels)/max(1,2*n_pos)])
    for rnd in range(R):
        deltas, n_k_list = [], []
        for k in range(len(client_train)):
            if len(client_train[k]) < 1: continue
            delta = local_train_fedavg(global_flat, client_train[k], model_template,
                                        local_epochs=local_epochs, class_weights=class_weights)
            deltas.append(delta); n_k_list.append(len(client_train[k]))
        n_total = sum(n_k_list)
        agg_delta = sum((n_k/n_total)*d for n_k, d in zip(n_k_list, deltas))
        global_flat = global_flat + agg_delta
    all_val = [s for cv in client_val for s in cv]
    return evaluate_fedavg(model_template, global_flat, all_val)


if __name__ == "__main__":
    all_results = {"FedGTCL-IID": [], "FedGTCL-nonIID": [],
                   "FedAvgGCNGRU-IID": [], "FedAvgGCNGRU-nonIID": []}

    import pickle
    for seed in SEEDS:
        print(f"\n{'='*30} SEED {seed} {'='*30}")
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

        # Use the CANONICAL, pre-shared partition for this seed (identical
        # across Tables 6, 11, 12, 13) rather than re-deriving one locally,
        # so that "seed=42" always means the exact same client/train/val
        # split everywhere it is cited in the manuscript.
        with open(f"data/processed/shared_partition_{seed}.pkl", "rb") as f:
            _P = pickle.load(f)
        iid_train, iid_val = _P["iid_train"], _P["iid_val"]
        noniid_train, noniid_val = _P["noniid_train"], _P["noniid_val"]

        r1 = run_fedgtcl(iid_train, iid_val); print(f"  FedGTCL IID: F1={r1['F1']:.2f}%")
        all_results["FedGTCL-IID"].append(r1)
        r2 = run_fedgtcl(noniid_train, noniid_val); print(f"  FedGTCL non-IID: F1={r2['F1']:.2f}%")
        all_results["FedGTCL-nonIID"].append(r2)
        r3 = run_fedavg_gcngru(iid_train, iid_val); print(f"  FedAvg-GCN-GRU IID: F1={r3['F1']:.2f}%")
        all_results["FedAvgGCNGRU-IID"].append(r3)
        r4 = run_fedavg_gcngru(noniid_train, noniid_val); print(f"  FedAvg-GCN-GRU non-IID: F1={r4['F1']:.2f}%")
        all_results["FedAvgGCNGRU-nonIID"].append(r4)

    print(f"\n\n{'='*30} MULTI-SEED SUMMARY (n={len(SEEDS)} seeds) {'='*30}")
    for key, runs in all_results.items():
        f1s = [r['F1'] for r in runs]
        accs = [r['Accuracy'] for r in runs]
        print(f"{key}: F1 = {np.mean(f1s):.2f}% +/- {np.std(f1s):.2f}  "
              f"(individual: {[round(x,2) for x in f1s]})  "
              f"Accuracy = {np.mean(accs):.2f}% +/- {np.std(accs):.2f}")
