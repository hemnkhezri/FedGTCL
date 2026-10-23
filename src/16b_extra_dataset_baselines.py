# Paper output: Online Resource 1, Table S7, baseline rows; usage: <xiiotid|cicapt>
# Run from the repository root.
"""
Step 16b: the three baselines of 09b_edgeiiot_baselines.py (Centralized GCN-GRU, FedAvg-GCN-GRU, FedAvg-LSTM)
on the shared partition saved by 16_extra_dataset_pipeline.py for X-IIoTID or CICAPT-IIoT2024.
"""
import pickle, random, copy
import numpy as np
import torch
import torch.nn.functional as F

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

import sys
DS = sys.argv[1]
with open(f"data/processed/shared_partition_{DS}.pkl", "rb") as f:
    P = pickle.load(f)

IN_DIM = P["IN_DIM"]; HIDDEN = 64; NUM_CLASSES = 2


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
            h = self.dense_gcn_layer(f, self.gcn1)
            h = self.dense_gcn_layer(h, self.gcn2)
            graph_embeds.append(h.mean(dim=0))
        seq = torch.stack(graph_embeds, dim=0).unsqueeze(0)
        out, _ = self.gru(seq)
        return self.cls(out[:, -1, :]).squeeze(0)


class LSTMBaseline(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes):
        super().__init__()
        self.proj = torch.nn.Linear(in_dim, hidden_dim)
        self.lstm = torch.nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.cls = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, feats_seq):
        pooled = [torch.relu(self.proj(f.mean(dim=0))) for f in feats_seq]
        seq = torch.stack(pooled, dim=0).unsqueeze(0)
        out, _ = self.lstm(seq)
        return self.cls(out[:, -1, :]).squeeze(0)


def flatten_params(model): return torch.cat([p.data.view(-1) for p in model.parameters()])
def load_flat_params(model, flat):
    i = 0
    for p in model.parameters():
        n = p.numel(); p.data.copy_(flat[i:i + n].view_as(p)); i += n


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
        loss = F.cross_entropy(torch.stack(logits_list, dim=0),
                                torch.tensor(labels_list, dtype=torch.long), weight=class_weights)
        loss.backward(); opt.step()
    return flatten_params(model) - global_flat


def evaluate_fedavg(model, flat_params, seqs):
    load_flat_params(model, flat_params)
    model.eval()
    TP = FP = TN = FN = 0
    with torch.no_grad():
        for entry in seqs:
            feats = [w["feats"] for w in entry["seq"]]
            logits = model(feats)
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


def run_fedavg(model_cls, name, client_train, client_val, R=10, local_epochs=3):
    model_template = model_cls(IN_DIM, HIDDEN, NUM_CLASSES)
    n_params = sum(p.numel() for p in model_template.parameters() if p.requires_grad)
    global_flat = flatten_params(model_template).clone()
    all_train_labels = [s["label"] for tr in client_train for s in tr]
    n_pos = sum(all_train_labels); n_neg = len(all_train_labels) - n_pos
    class_weights = torch.tensor([len(all_train_labels) / max(1, 2 * n_neg),
                                    len(all_train_labels) / max(1, 2 * n_pos)])
    print(f"\n--- {model_cls.__name__}: {name} (params={n_params}) ---")
    for rnd in range(R):
        deltas, n_k_list = [], []
        for k in range(len(client_train)):
            if len(client_train[k]) < 1: continue
            delta = local_train_fedavg(global_flat, client_train[k], model_template,
                                        local_epochs=local_epochs, class_weights=class_weights)
            deltas.append(delta); n_k_list.append(len(client_train[k]))
        n_total = sum(n_k_list)
        agg_delta = sum((n_k / n_total) * d for n_k, d in zip(n_k_list, deltas))
        global_flat = global_flat + agg_delta
        if (rnd + 1) % 5 == 0 or rnd == R - 1:
            all_val = [s for cv in client_val for s in cv]
            m = evaluate_fedavg(model_template, global_flat, all_val)
            print(f"  round {rnd+1}/{R}: Accuracy={m['Accuracy']:.2f}% F1={m['F1']:.2f}% Recall={m['Recall']:.2f}%")
    all_val = [s for cv in client_val for s in cv]
    return evaluate_fedavg(model_template, global_flat, all_val), n_params


def evaluate_centralized(model, seqs):
    model.eval()
    TP = FP = TN = FN = 0
    with torch.no_grad():
        for entry in seqs:
            feats = [w["feats"] for w in entry["seq"]]
            logits = model(feats)
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


def run_centralized_gaco(all_sequences, epochs=40, batch_size=16, lr=1e-3):
    idxs = list(range(len(all_sequences))); random.shuffle(idxs)
    cut = int(0.8 * len(idxs))
    train_seqs = [all_sequences[i] for i in idxs[:cut]]
    val_seqs = [all_sequences[i] for i in idxs[cut:]]
    n_pos = sum(s["label"] for s in train_seqs); n_neg = len(train_seqs) - n_pos
    class_weights = torch.tensor([len(train_seqs) / max(1, 2 * n_neg),
                                    len(train_seqs) / max(1, 2 * n_pos)])
    print(f"\n--- Centralized GACO (train={len(train_seqs)}, val={len(val_seqs)}) ---")
    model = GCNGRUBaseline(IN_DIM, HIDDEN, NUM_CLASSES)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train(); random.shuffle(train_seqs)
        for b in range(0, len(train_seqs), batch_size):
            batch = train_seqs[b:b + batch_size]
            opt.zero_grad()
            logits_list, labels_list = [], []
            for entry in batch:
                feats = [w["feats"] for w in entry["seq"]]
                logits_list.append(model(feats)); labels_list.append(entry["label"])
            loss = F.cross_entropy(torch.stack(logits_list, dim=0),
                                    torch.tensor(labels_list, dtype=torch.long), weight=class_weights)
            loss.backward(); opt.step()
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            m = evaluate_centralized(model, val_seqs)
            print(f"  epoch {epoch+1}/{epochs}: Accuracy={m['Accuracy']:.2f}% F1={m['F1']:.2f}% Recall={m['Recall']:.2f}%")
    return evaluate_centralized(model, val_seqs), n_params


if __name__ == "__main__":
    results = {}
    results["Centralized-GACO"], n_p1 = run_centralized_gaco(P["all_sequences"], epochs=40)
    results["FedAvg-GCNGRU-IID"], n_p2 = run_fedavg(GCNGRUBaseline, "IID", P["iid_train"], P["iid_val"])
    results["FedAvg-GCNGRU-nonIID"], _ = run_fedavg(GCNGRUBaseline, "non-IID", P["noniid_train"], P["noniid_val"])
    results["FedAvg-LSTM-IID"], n_p3 = run_fedavg(LSTMBaseline, "IID", P["iid_train"], P["iid_val"])
    results["FedAvg-LSTM-nonIID"], _ = run_fedavg(LSTMBaseline, "non-IID", P["noniid_train"], P["noniid_val"])

    print(f"\n\n=== {DS} BASELINES FINAL RESULTS (params: GACO/GCNGRU={n_p2}, LSTM={n_p3}) ===")
    for k, v in results.items():
        print(f"{k}: {v}")
