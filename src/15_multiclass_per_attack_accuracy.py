# Paper output: Fig. 12 and Section 4.14 (per-attack-type accuracy)
# Run from the repository root.
"""
Step 15: Real per-attack-type detection accuracy, analogous to base paper
Fig. 11, computed on real data across all three datasets. Unlike the
window-level graph classification used elsewhere in this study (which is
necessarily binary here, since majority-vote window labels collapse rare
attack types away -- Section 5, limitation iv), this experiment classifies
INDIVIDUAL FLOWS directly on their normalized tabular features with a
simple centralized MLP, to obtain genuine multi-class, per-attack-type
accuracy without that collapse. This is a deliberately simpler, flow-level,
centralized setup -- not federated, not graph-based -- used ONLY to produce
a real multi-class breakdown; it is reported as such, not conflated with
the federated window-level results elsewhere in this study.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)
np.random.seed(42)

DATASETS = {
    "WUSTL-IIoT-2021": {
        "path": "data/processed/WUSTL-IIoT-2021_processed.csv",
        "reserved": {"StartTime", "LastTime", "SrcAddr", "DstAddr", "Sport", "Dport", "Target", "Traffic"},
        "type_col": "Traffic",
    },
    "TON_IoT-Network": {
        "path": "data/processed/TON_IoT-Network_processed.csv",
        "reserved": {"src_ip", "src_port", "dst_ip", "dst_port", "label", "type"},
        "type_col": "type",
    },
    "EdgeIIoTset-10pct": {
        "path": "data/processed/EdgeIIoTset-10pct_processed.csv",
        "reserved": {"ip.src_host", "ip.dst_host", "tcp.srcport", "tcp.dstport", "udp.port",
                     "frame.time", "Attack_label", "Attack_type"},
        "type_col": "Attack_type",
    },
}

class MLPClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )
    def forward(self, x):
        return self.net(x)

all_results = {}

for name, cfg in DATASETS.items():
    print(f"\n{'='*60}\n{name}\n{'='*60}")
    df = pd.read_csv(cfg["path"])
    type_col = cfg["type_col"]
    feature_cols = [c for c in df.columns if c not in cfg["reserved"]]

    classes = sorted(df[type_col].unique().tolist())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y = df[type_col].map(class_to_idx).values
    X = df[feature_cols].values.astype(np.float32)

    # stratified 80/20 split
    idx = np.arange(len(df))
    rng = np.random.RandomState(42)
    rng.shuffle(idx)
    train_idx_list, test_idx_list = [], []
    for c in range(len(classes)):
        c_idx = idx[y[idx] == c]
        cut = max(1, int(0.8 * len(c_idx)))
        train_idx_list.append(c_idx[:cut])
        test_idx_list.append(c_idx[cut:])
    train_idx = np.concatenate(train_idx_list)
    test_idx = np.concatenate(test_idx_list)

    X_train = torch.tensor(X[train_idx]); y_train = torch.tensor(y[train_idx], dtype=torch.long)
    X_test = torch.tensor(X[test_idx]); y_test = torch.tensor(y[test_idx], dtype=torch.long)

    # class-balanced weights (severe imbalance across attack types)
    class_counts = np.bincount(y_train.numpy(), minlength=len(classes))
    class_weights = torch.tensor(len(y_train) / (len(classes) * np.maximum(class_counts, 1)), dtype=torch.float32)

    model = MLPClassifier(len(feature_cols), 64, len(classes))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    BATCH = 512
    EPOCHS = 15
    n_train = len(X_train)
    for epoch in range(EPOCHS):
        perm = torch.randperm(n_train)
        model.train()
        total_loss = 0.0
        for b in range(0, n_train, BATCH):
            bidx = perm[b:b + BATCH]
            opt.zero_grad()
            logits = model(X_train[bidx])
            loss = F.cross_entropy(logits, y_train[bidx], weight=class_weights)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(bidx)
        if (epoch + 1) % 5 == 0:
            print(f"  epoch {epoch+1}/{EPOCHS}: train_loss={total_loss/n_train:.4f}")

    model.eval()
    with torch.no_grad():
        logits = model(X_test)
        preds = logits.argmax(dim=1)

    per_class_acc = {}
    for c, idx_name in class_to_idx.items():
        mask = (y_test == idx_name)
        if mask.sum() > 0:
            acc = (preds[mask] == y_test[mask]).float().mean().item() * 100
            per_class_acc[c] = (acc, int(mask.sum()))

    overall_acc = (preds == y_test).float().mean().item() * 100
    print(f"  Overall test accuracy: {overall_acc:.2f}%")
    print("  Per-class accuracy (class: acc%, n_test):")
    for c, (acc, n) in sorted(per_class_acc.items(), key=lambda x: -x[1][1]):
        print(f"    {c:<25} {acc:6.2f}%  (n={n})")

    all_results[name] = per_class_acc

print("\n\n=== FULL PER-ATTACK-TYPE RESULTS (real, for Fig. 11 equivalent) ===")
for name, res in all_results.items():
    print(f"\n{name}:")
    for c, (acc, n) in sorted(res.items(), key=lambda x: -x[1][1]):
        print(f"  {c!r}: {acc:.2f}")
