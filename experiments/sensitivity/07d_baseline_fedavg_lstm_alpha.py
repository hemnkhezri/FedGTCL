# Paper output: Online Resource 1, Table S1 (FedAvg-LSTM); usage: <seed> <alpha>
# Run from the repository root.
"""
Step 7d: FedAvg-LSTM baseline (Section 4.4, baseline #3) -- a non-graph
recurrent baseline representative of generic federated IDS approaches
[8,12] that do NOT model device-interaction structure at all: each window
is collapsed to a single flat mean-pooled feature vector (no adjacency,
no attention, no per-neighbor aggregation whatsoever), and the T-length
sequence of these vectors is processed by a standard LSTM. Trained under
plain FedAvg on the IDENTICAL shared partition as FedGTCL and FedAvg-GCN-GRU.
"""
import pickle, random, copy
import numpy as np
import torch
import torch.nn.functional as F

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

import sys
_PARTSEED = sys.argv[1] if len(sys.argv)>1 else "42"
import os
_ALPHA = sys.argv[2] if len(sys.argv) > 2 else None
_fname = f"data/processed/shared_partition_{_PARTSEED}_a{_ALPHA}.pkl" if _ALPHA else f"data/processed/shared_partition_{_PARTSEED}.pkl"
with open(_fname, "rb") as f:
    P = pickle.load(f)

IN_DIM = P["IN_DIM"]
HIDDEN = 64
NUM_CLASSES = 2


class LSTMBaseline(torch.nn.Module):
    """No graph structure at all: each window -> flat mean-pooled feature
    vector -> LSTM over the T-length sequence -> classification head."""
    def __init__(self, in_dim, hidden_dim, num_classes):
        super().__init__()
        self.proj = torch.nn.Linear(in_dim, hidden_dim)
        self.lstm = torch.nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.cls = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, feats_seq):
        # feats_seq: list of T tensors, each [N_t, in_dim] -> mean-pool per window
        pooled = [torch.relu(self.proj(f.mean(dim=0))) for f in feats_seq]  # T x [hidden]
        seq = torch.stack(pooled, dim=0).unsqueeze(0)  # [1, T, hidden]
        out, _ = self.lstm(seq)
        return self.cls(out[:, -1, :]).squeeze(0)


def flatten_params(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()])

def load_flat_params(model, flat):
    i = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat[i:i + n].view_as(p))
        i += n


def local_train_fedavg(global_flat, client_seqs, model_template,
                        local_epochs=3, lr=1e-3, class_weights=None):
    model = copy.deepcopy(model_template)
    load_flat_params(model, global_flat)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

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
            logits_list.append(model(feats))
            labels_list.append(entry["label"])
        logits_batch = torch.stack(logits_list, dim=0)
        labels_batch = torch.tensor(labels_list, dtype=torch.long)
        loss = F.cross_entropy(logits_batch, labels_batch, weight=class_weights)
        loss.backward()
        opt.step()

    return flatten_params(model) - global_flat


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


def run_fedavg(name, client_train, client_val, R=10, local_epochs=3):
    model_template = LSTMBaseline(IN_DIM, HIDDEN, NUM_CLASSES)
    n_params = sum(p.numel() for p in model_template.parameters() if p.requires_grad)
    global_flat = flatten_params(model_template).clone()

    all_train_labels = [s["label"] for tr in client_train for s in tr]
    n_pos = sum(all_train_labels); n_neg = len(all_train_labels) - n_pos
    class_weights = torch.tensor([len(all_train_labels) / max(1, 2 * n_neg),
                                    len(all_train_labels) / max(1, 2 * n_pos)])

    print(f"\n--- FedAvg-LSTM: {name} (params={n_params}) ---")
    for rnd in range(R):
        deltas, n_k_list = [], []
        for k in range(len(client_train)):
            if len(client_train[k]) < 1:
                continue
            delta = local_train_fedavg(global_flat, client_train[k], model_template,
                                        local_epochs=local_epochs, class_weights=class_weights)
            deltas.append(delta)
            n_k_list.append(len(client_train[k]))
        n_total = sum(n_k_list)
        agg_delta = sum((n_k / n_total) * d for n_k, d in zip(n_k_list, deltas))
        global_flat = global_flat + agg_delta

        if (rnd + 1) % 3 == 0 or rnd == R - 1:
            all_val = [s for cv in client_val for s in cv]
            m = evaluate_fedavg(model_template, global_flat, all_val)
            print(f"  round {rnd+1}/{R}: Accuracy={m['Accuracy']:.2f}% F1={m['F1']:.2f}% "
                  f"Recall={m['Recall']:.2f}% FNR={m['FNR']:.2f}%")

    all_val = [s for cv in client_val for s in cv]
    return evaluate_fedavg(model_template, global_flat, all_val), n_params


if __name__ == "__main__":
    results = {}
    results["IID"], n_params = run_fedavg("IID", P["iid_train"], P["iid_val"], R=10, local_epochs=3)
    results["non-IID"], _ = run_fedavg("non-IID", P["noniid_train"], P["noniid_val"], R=10, local_epochs=3)

    print(f"\n\n=== FedAvg-LSTM FINAL RESULTS (real, WUSTL-IIoT-2021, params={n_params}) ===")
    for k, v in results.items():
        print(f"{k}: {v}")
