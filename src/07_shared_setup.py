# Paper output: Section 4.3: canonical seed-42 IID / non-IID partition shared by all methods
# Run from the repository root.
"""
Step 7a: Shared setup -- build the real WUSTL graph windows/sequences ONCE
and save the exact IID / non-IID partition to disk, so that FedGTCL and
every baseline are trained/evaluated on IDENTICAL client shards (fair
comparison; Section 4.3-4.4).
"""
import importlib.util, random, pickle
import numpy as np
import pandas as pd
import torch

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

graph_mod = load_module("graph_mod", "src/03_graph_construction.py")

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

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
        wdf, "SrcAddr", "DstAddr", feature_cols, top_m=TOP_M
    )
    label = int(wdf["Target"].max())
    windows_data.append({
        "id": w, "node_ids": node_ids,
        "feats": torch.tensor(node_feats, dtype=torch.float32),
        "mask": torch.tensor(adj_mask, dtype=torch.bool),
        "label": label,
    })

T = 5
def make_sequences(win_list):
    seqs = []
    for i in range(T - 1, len(win_list)):
        seq = win_list[i - T + 1: i + 1]
        seqs.append({"seq": seq, "label": win_list[i]["label"], "anchor_idx": i})
    return seqs

all_sequences = make_sequences(windows_data)
K = 4

def partition_iid(sequences, k):
    idx = list(range(len(sequences)))
    random.shuffle(idx)
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
            print(f"Dirichlet partition accepted after {attempt+1} attempt(s), sizes={sizes}")
            return client_idx
    return client_idx

iid_clients = partition_iid(all_sequences, K)
noniid_clients = partition_noniid_dirichlet(all_sequences, K, alpha=0.3)

print("IID sizes:", [len(c) for c in iid_clients])
print("non-IID sizes:", [len(c) for c in noniid_clients])

# 80/20 train/val split per client -- fixed once, reused by every method
def make_train_val(client_idx_lists, sequences):
    train, val = [], []
    for idxs in client_idx_lists:
        idxs2 = idxs.copy(); random.shuffle(idxs2)
        cut = max(1, int(0.8 * len(idxs2)))
        train.append([sequences[i] for i in idxs2[:cut]])
        val.append([sequences[i] for i in idxs2[cut:]])
    return train, val

iid_train, iid_val = make_train_val(iid_clients, all_sequences)
noniid_train, noniid_val = make_train_val(noniid_clients, all_sequences)

with open("data/processed/shared_partition.pkl", "wb") as f:
    pickle.dump({
        "windows_data": windows_data,
        "all_sequences": all_sequences,
        "IN_DIM": IN_DIM,
        "T": T,
        "iid_train": iid_train, "iid_val": iid_val,
        "noniid_train": noniid_train, "noniid_val": noniid_val,
    }, f)

print("\nSaved shared partition -> data/processed/shared_partition.pkl")
print(f"IID   train sizes: {[len(t) for t in iid_train]}  val sizes: {[len(v) for v in iid_val]}")
print(f"non-IID train sizes: {[len(t) for t in noniid_train]}  val sizes: {[len(v) for v in noniid_val]}")
