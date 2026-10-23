# Paper output: Online Resource 1, Table S6 (SCAFFOLD)
# Run from the repository root.
"""
Step 7b: FedAvg-GCN-GRU baseline (Section 4.4, baseline #2), trained and
evaluated on the EXACT SAME client partition as FedGTCL (loaded from
shared_partition.pkl), using plain federated averaging (no momentum
correction, no sparsification, no contrastive pretraining, no low-rank
projection, full dense adjacency) -- isolating the effect of FedGTCL's
architectural and optimization choices.
"""
import pickle, random, copy
import numpy as np
import torch
import torch.nn.functional as F

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

with open("data/processed/shared_partition.pkl", "rb") as f:
    P = pickle.load(f)

IN_DIM = P["IN_DIM"]
T = P["T"]
HIDDEN = 64
NUM_CLASSES = 2


class GCNGRUBaseline(torch.nn.Module):
    """Full-rank, full-adjacency GCN + real GRU (Section 4.4 baseline #2),
    the non-sparsified, non-low-rank, non-contrastive counterpart of the
    FedGTCL encoder -- structurally the closest reproduction of GACO's
    GCN-GRU [28] design achievable without the original codebase."""
    def __init__(self, in_dim, hidden_dim, num_classes):
        super().__init__()
        self.gcn1 = torch.nn.Linear(in_dim, hidden_dim)
        self.gcn2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.gru = torch.nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.cls = torch.nn.Linear(hidden_dim, num_classes)

    def dense_gcn_layer(self, h, lin):
        N = h.size(0)
        Wh = lin(h)
        A_full = torch.ones(N, N) / N
        return torch.relu(A_full @ Wh)

    def forward(self, feats_seq):
        graph_embeds = []
        for f in feats_seq:
            h = self.dense_gcn_layer(f, self.gcn1)
            h = self.dense_gcn_layer(h, self.gcn2)
            graph_embeds.append(h.mean(dim=0))
        seq = torch.stack(graph_embeds, dim=0).unsqueeze(0)
        out, _ = self.gru(seq)
        return self.cls(out[:, -1, :]).squeeze(0)


def flatten_params(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()])

def load_flat_params(model, flat):
    i = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat[i:i + n].view_as(p))
        i += n


def local_train_scaffold(global_flat, client_seqs, model_template,
                          local_epochs=3, lr=1e-3, class_weights=None,
                          c_global=None, c_local=None):
    """SCAFFOLD (Karimireddy et al. 2020): identical to plain FedAvg local
    training, plus a control-variate correction term added to the loss whose
    gradient is exactly (c_global - c_local), directly counteracting client
    drift -- the correction is applied as a linear term in the loss so it
    composes with the Adam optimizer already used for local steps."""
    model = copy.deepcopy(model_template)
    load_flat_params(model, global_flat)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    correction = (c_global - c_local).detach()  # fixed per round, per SCAFFOLD Option II

    MAX_BATCH = 16
    for epoch in range(local_epochs):
        random.shuffle(client_seqs)
        batch = client_seqs[:MAX_BATCH]
        if len(batch) < 1:
            continue
        opt.zero_grad()
        logits_list, labels_list = [], []
        for entry in batch:
            feats = [w["feats"] for w in entry["seq"]]
            logits = model(feats)
            logits_list.append(logits)
            labels_list.append(entry["label"])
        logits_batch = torch.stack(logits_list, dim=0)
        labels_batch = torch.tensor(labels_list, dtype=torch.long)
        ce_loss = F.cross_entropy(logits_batch, labels_batch, weight=class_weights)
        # linear control-variate term: grad w.r.t. params is exactly `correction`
        offset = 0
        corr_term = 0.0
        for p in model.parameters():
            n = p.numel()
            corr_term = corr_term + torch.sum(correction[offset:offset + n].view_as(p) * p)
            offset += n
        loss = ce_loss + corr_term
        loss.backward()
        opt.step()

    local_flat = flatten_params(model)
    delta_y = local_flat - global_flat
    # Option II control-variate update
    c_local_new = c_local - c_global - (1.0 / (local_epochs * lr)) * delta_y
    return delta_y, (c_local_new - c_local).detach()


def evaluate_fedavg(model, flat_params, seqs):
    load_flat_params(model, flat_params)
    model.eval()
    TP = FP = TN = FN = 0
    with torch.no_grad():
        for entry in seqs:
            feats = [w["feats"] for w in entry["seq"]]
            logits = model(feats)
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


def run_scaffold(name, client_train, client_val, R=10, local_epochs=3, lr=1e-3):
    model_template = GCNGRUBaseline(IN_DIM, HIDDEN, NUM_CLASSES)
    n_params = sum(p.numel() for p in model_template.parameters() if p.requires_grad)
    global_flat = flatten_params(model_template).clone()
    K = len(client_train)
    c_global = torch.zeros_like(global_flat)
    c_locals = [torch.zeros_like(global_flat) for _ in range(K)]

    all_train_labels = [s["label"] for tr in client_train for s in tr]
    n_pos = sum(all_train_labels); n_neg = len(all_train_labels) - n_pos
    class_weights = torch.tensor([len(all_train_labels) / max(1, 2 * n_neg),
                                    len(all_train_labels) / max(1, 2 * n_pos)])

    print(f"\n--- SCAFFOLD-GCN-GRU: {name} (params={n_params}) ---")
    for rnd in range(R):
        deltas, delta_cs, n_k_list = [], [], []
        for k in range(K):
            if len(client_train[k]) < 1:
                continue
            delta_y, delta_c = local_train_scaffold(
                global_flat, client_train[k], model_template,
                local_epochs=local_epochs, lr=lr, class_weights=class_weights,
                c_global=c_global, c_local=c_locals[k])
            deltas.append(delta_y)
            delta_cs.append(delta_c)
            c_locals[k] = c_locals[k] + delta_c
            n_k_list.append(len(client_train[k]))
        n_total = sum(n_k_list)
        agg_delta = sum((n_k / n_total) * d for n_k, d in zip(n_k_list, deltas))
        global_flat = global_flat + agg_delta
        # Option II global control variate update: uniform average over all K clients
        c_global = c_global + (1.0 / K) * sum(delta_cs)

        if (rnd + 1) % 3 == 0 or rnd == R - 1:
            all_val = [s for cv in client_val for s in cv]
            m = evaluate_fedavg(model_template, global_flat, all_val)
            print(f"  round {rnd+1}/{R}: Accuracy={m['Accuracy']:.2f}% F1={m['F1']:.2f}% "
                  f"Recall={m['Recall']:.2f}% FNR={m['FNR']:.2f}%")

    all_val = [s for cv in client_val for s in cv]
    return evaluate_fedavg(model_template, global_flat, all_val), n_params


if __name__ == "__main__":
    results = {}
    results["IID"], n_params = run_scaffold("IID", P["iid_train"], P["iid_val"], R=10, local_epochs=3)
    results["non-IID"], _ = run_scaffold("non-IID", P["noniid_train"], P["noniid_val"], R=10, local_epochs=3)

    print(f"\n\n=== SCAFFOLD-GCN-GRU FINAL RESULTS (real, WUSTL-IIoT-2021, params={n_params}) ===")
    for k, v in results.items():
        print(f"{k}: {v}")
