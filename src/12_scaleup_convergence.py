# Paper output: Table 13 and Fig. 11 (R=30 checkpoints; needs shared_partition_42.pkl)
# Run from the repository root.
"""
Step 12: Scale-up convergence check -- repeats the core WUSTL-IIoT-2021
FedGTCL comparison at R=30 (tripled from the R=10 pilot), logging F1 every
3 rounds, to test whether the IID/non-IID gap and the FedGTCL/baseline gap
observed in the R=10 pilot (Sections 5.1, 5.6) narrow with more
communication rounds -- directly addressing the "pilot scale" limitation
flagged in Section 6.4.
"""
import importlib.util, random, copy
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

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

df = pd.read_csv("data/processed/WUSTL-IIoT-2021_processed.csv")
reserved = {"StartTime", "LastTime", "SrcAddr", "DstAddr", "Sport", "Dport", "Target", "Traffic"}
feature_cols = [c for c in df.columns if c not in reserved]
IN_DIM = len(feature_cols)
df = graph_mod.make_time_windows(df, "StartTime", window_seconds=60)
window_ids = sorted(df["window_id"].unique())

TOP_M = 8
import pickle
with open("data/processed/shared_partition_42.pkl", "rb") as f:
    _P = pickle.load(f)
windows_data = _P["windows_data"]
IN_DIM = _P["IN_DIM"]
T = _P["T"]
iid_train, iid_val = _P["iid_train"], _P["iid_val"]
noniid_train, noniid_val = _P["noniid_train"], _P["noniid_val"]
print(f"Loaded canonical shared_partition_42.pkl: IID val sizes {[len(v) for v in iid_val]}, "
      f"non-IID val sizes {[len(v) for v in noniid_val]}")
def augment_edge_dropout(mask, rho=0.2):
    mask = mask.clone()
    drop = torch.rand_like(mask, dtype=torch.float32) < rho
    mask = mask & ~drop
    isolated = ~mask.any(dim=1)
    if isolated.any(): mask[isolated, isolated] = True
    return mask

def build_view(seq_entry, augment=True):
    seq = seq_entry["seq"]
    feats = [w["feats"] for w in seq]
    masks = [augment_edge_dropout(w["mask"], rho=0.2) if augment else w["mask"] for w in seq]
    return feats, masks

def structure_aware_nt_xent(z_list, tau=0.5):
    B2 = len(z_list); Z = torch.stack(z_list, dim=0); sim = (Z @ Z.t()) / tau
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
        n = p.numel(); p.data.copy_(flat[i:i+n].view_as(p)); i += n
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
    TP=FP=TN=FN=0
    with torch.no_grad():
        for entry in seqs:
            feats, masks = build_view(entry, augment=False)
            _, _, logits = model(feats, masks)
            pred = int(logits.argmax()); y = entry["label"]
            if pred==1 and y==1: TP+=1
            elif pred==1 and y==0: FP+=1
            elif pred==0 and y==0: TN+=1
            elif pred==0 and y==1: FN+=1
    acc=100*(TP+TN)/max(1,TP+TN+FP+FN); prec=100*TP/max(1,TP+FP)
    rec=100*TP/max(1,TP+FN); f1=2*prec*rec/max(1e-8,prec+rec)
    return dict(Accuracy=acc, Precision=prec, Recall=rec, F1=f1, TP=TP,FP=FP,TN=TN,FN=FN)

def run_fedgtcl_scaleup(name, client_train, client_val, R=30, local_epochs=3, s=0.3, eta_s=0.02,
                          beta1=0.9, beta2=0.99, p_label=0.5, log_every=5):
    model_template = model_mod.FedGTCLEncoder(in_dim=IN_DIM, hidden_dim=64, embed_dim=64,
                                                low_rank_dim=32, num_classes=2, proj_dim=32, temporal_kernel=3)
    global_flat = flatten_params(model_template).clone()
    m_t = torch.zeros_like(global_flat); v_t = torch.zeros_like(global_flat); eps = 1e-8
    all_train_labels = [s2["label"] for tr in client_train for s2 in tr]
    n_pos = sum(all_train_labels); n_neg = len(all_train_labels)-n_pos
    class_weights = torch.tensor([len(all_train_labels)/max(1,2*n_neg), len(all_train_labels)/max(1,2*n_pos)])
    history = []
    print(f"\n--- FedGTCL scale-up: {name} (R={R}) ---")
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
        if (rnd+1) % log_every == 0 or rnd == R-1:
            all_val = [s2 for cv in client_val for s2 in cv]
            m = evaluate_fedgtcl(model_template, global_flat, all_val)
            print(f"  round {rnd+1}/{R}: F1={m['F1']:.2f}% Acc={m['Accuracy']:.2f}% Recall={m['Recall']:.2f}%")
            history.append((rnd+1, m['F1']))
    return history


if __name__ == "__main__":
    hist_iid = run_fedgtcl_scaleup("IID", iid_train, iid_val, R=30, local_epochs=3, log_every=5)
    hist_noniid = run_fedgtcl_scaleup("non-IID", noniid_train, noniid_val, R=30, local_epochs=3, log_every=5)

    print("\n\n=== SCALE-UP SUMMARY (R=10 vs R=30) ===")
    print("IID history (round, F1%):", hist_iid)
    print("non-IID history (round, F1%):", hist_noniid)
