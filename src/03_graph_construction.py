# Paper output: Section 3.2 (module: windowing and host-level graph construction)
# Run from the repository root.
"""
Step 3: Graph construction from IIoT traffic (Section 3.2 of the manuscript).

Design (as implemented, refined from the initial (IP,port) node definition
described in the text draft): each NODE is a communicating HOST, identified
by IP address alone. Ports and protocol become EDGE-level aggregated
attributes, not part of node identity. This avoids node-count blow-up from
ephemeral source ports and matches the "device-interaction graph" framing
used throughout the manuscript.
"""
import pandas as pd
import numpy as np


def build_window_graph(window_df, src_col, dst_col, feature_cols, top_m=8):
    """
    Build one device-interaction graph from all flow rows in a single time
    window.

    Returns:
        node_ids   : list of host identifiers (str), length N
        node_feats : np.ndarray [N, d]  (mean of numeric feature_cols over
                     all flows where the host appears as src or dst)
        adj_mask   : np.ndarray [N, N] bool, True where v is in u's
                     top-m sparsified neighborhood (Section 3.3a)
        edge_weight: np.ndarray [N, N] float, aggregated flow count per pair
    """
    hosts = pd.unique(pd.concat([window_df[src_col], window_df[dst_col]]))
    node_ids = sorted(hosts.tolist())
    idx = {h: i for i, h in enumerate(node_ids)}
    N = len(node_ids)
    d = len(feature_cols)

    # --- node features: mean of numeric features over flows touching host ---
    node_feats = np.zeros((N, d), dtype=np.float32)
    node_counts = np.zeros(N, dtype=np.int64)

    feat_matrix = window_df[feature_cols].values.astype(np.float32)
    src_idx = window_df[src_col].map(idx).values
    dst_idx = window_df[dst_col].map(idx).values

    for i in range(len(window_df)):
        s, t = src_idx[i], dst_idx[i]
        node_feats[s] += feat_matrix[i]
        node_counts[s] += 1
        node_feats[t] += feat_matrix[i]
        node_counts[t] += 1
    node_counts[node_counts == 0] = 1
    node_feats = node_feats / node_counts[:, None]

    # --- edge weights: aggregated flow count per (src,dst) pair ---
    edge_weight = np.zeros((N, N), dtype=np.float32)
    for s, t in zip(src_idx, dst_idx):
        edge_weight[s, t] += 1.0
        edge_weight[t, s] += 1.0  # treat as undirected for neighborhood purposes

    # --- top-m sparsification per node (Section 3.3a) ---
    adj_mask = np.zeros((N, N), dtype=bool)
    for v in range(N):
        row = edge_weight[v].copy()
        row[v] = -1  # exclude self
        m_eff = min(top_m, (row > 0).sum())
        if m_eff > 0:
            top_idx = np.argpartition(-row, m_eff - 1)[:m_eff]
            top_idx = top_idx[row[top_idx] > 0]
            adj_mask[v, top_idx] = True

    return node_ids, node_feats, adj_mask, edge_weight


def make_time_windows(df, time_col, window_seconds=60):
    """Assign each row a window_id based on a fixed-duration time bucket."""
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col).reset_index(drop=True)
    t0 = df[time_col].min()
    df["window_id"] = ((df[time_col] - t0).dt.total_seconds() // window_seconds).astype(int)
    return df


if __name__ == "__main__":
    df = pd.read_csv("data/processed/WUSTL-IIoT-2021_processed.csv")
    reserved = {"StartTime", "LastTime", "SrcAddr", "DstAddr", "Sport", "Dport", "Target", "Traffic"}
    feature_cols = [c for c in df.columns if c not in reserved]
    print(f"Using {len(feature_cols)} numeric node/edge features")

    df = make_time_windows(df, "StartTime", window_seconds=60)
    n_windows = df["window_id"].nunique()
    print(f"Total 60-second windows: {n_windows}")

    # build graphs for the first 8 windows as a smoke test
    example_windows = sorted(df["window_id"].unique())[:8]
    graphs = []
    for w in example_windows:
        wdf = df[df["window_id"] == w]
        node_ids, node_feats, adj_mask, edge_weight = build_window_graph(
            wdf, "SrcAddr", "DstAddr", feature_cols, top_m=8
        )
        graphs.append((node_ids, node_feats, adj_mask, edge_weight))
        print(f"window {w}: N={len(node_ids)} nodes, "
              f"avg_degree={adj_mask.sum(axis=1).mean():.2f}, "
              f"feat_shape={node_feats.shape}")

    np.save("data/processed/example_graphs_meta.npy", np.array(len(graphs)))
    print("\nSmoke test complete: graph construction pipeline runs on real data.")
