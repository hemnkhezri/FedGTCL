# Paper output: Table 8 (Centralized GCN-GRU upper bound)
# Run from the repository root.
"""
Step 7c: Centralized GACO baseline (Section 4.4, baseline #1) -- the same
GCN-GRU architecture as the FedAvg-GCN-GRU baseline, but trained CENTRALLY
on the pooled dataset with full label access (no federated partitioning, no
partial-labeling restriction), representing the centralized upper-bound
reference against which the cost of decentralization (Section 3.1) is
measured.
"""
import pickle, random
import numpy as np
import torch
import torch.nn.functional as F

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

with open("data/processed/shared_partition.pkl", "rb") as f:
    P = pickle.load(f)

all_sequences = P["all_sequences"]
IN_DIM = P["IN_DIM"]
HIDDEN = 64
NUM_CLASSES = 2
EPOCHS = 40
BATCH_SIZE = 16
LR = 1e-3


class GCNGRUBaseline(torch.nn.Module):
    """Identical architecture to the FedAvg-GCN-GRU baseline (07b), so that
    the ONLY difference between this and the federated baseline is the
    training protocol (centralized, full-label vs. federated, partial-label)
    -- isolating the cost of decentralization specifically."""
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


def evaluate(model, seqs):
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


if __name__ == "__main__":
    # single centralized 80/20 split over the FULL pooled sequence set
    idxs = list(range(len(all_sequences)))
    random.shuffle(idxs)
    cut = int(0.8 * len(idxs))
    train_seqs = [all_sequences[i] for i in idxs[:cut]]
    val_seqs = [all_sequences[i] for i in idxs[cut:]]

    n_pos = sum(s["label"] for s in train_seqs)
    n_neg = len(train_seqs) - n_pos
    class_weights = torch.tensor([len(train_seqs) / max(1, 2 * n_neg),
                                    len(train_seqs) / max(1, 2 * n_pos)])
    print(f"Centralized pooled train={len(train_seqs)} (attack={n_pos}), val={len(val_seqs)}")
    print(f"class weights (benign, attack): {class_weights.tolist()}")

    model = GCNGRUBaseline(IN_DIM, HIDDEN, NUM_CLASSES)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params}")
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        random.shuffle(train_seqs)
        total_loss = 0.0
        for b in range(0, len(train_seqs), BATCH_SIZE):
            batch = train_seqs[b:b + BATCH_SIZE]
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
            total_loss += loss.item() * len(batch)

        if (epoch + 1) % 5 == 0 or epoch == EPOCHS - 1:
            m = evaluate(model, val_seqs)
            print(f"epoch {epoch+1}/{EPOCHS}: train_loss={total_loss/len(train_seqs):.4f}  "
                  f"val Accuracy={m['Accuracy']:.2f}% F1={m['F1']:.2f}% "
                  f"Recall={m['Recall']:.2f}% FNR={m['FNR']:.2f}%")

    final = evaluate(model, val_seqs)
    print(f"\n\n=== Centralized GACO FINAL RESULTS (real, WUSTL-IIoT-2021, "
          f"params={n_params}, epochs={EPOCHS}) ===")
    print(final)
