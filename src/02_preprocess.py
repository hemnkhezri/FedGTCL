# Paper output: Table 3 and Section 4.1 dataset statistics
# Run from the repository root.
"""
Step 2: Unified preprocessing pipeline for WUSTL-IIoT-2021, TON_IoT-Network,
and Edge-IIoTset (10% stratified sample).
"""
import pandas as pd
import numpy as np
import os

os.makedirs("data/processed", exist_ok=True)
os.makedirs("outputs/tables", exist_ok=True)

HIGH_CARDINALITY_THRESHOLD = 50

DATASETS = {
    "WUSTL-IIoT-2021": {
        "path": "data/raw/wustl_iiot_2021.csv",
        "id_cols": ["SrcAddr", "DstAddr", "Sport", "Dport"],
        "time_cols": ["StartTime", "LastTime"],
        "label_col": "Target",
        "type_col": "Traffic",
    },
    "TON_IoT-Network": {
        "path": "data/raw/ton_iot_network.csv",
        "id_cols": ["src_ip", "dst_ip", "src_port", "dst_port"],
        "time_cols": [],
        "label_col": "label",
        "type_col": "type",
    },
    "EdgeIIoTset-10pct": {
        "path": "data/raw/edgeiiot_sample2_200k.csv",
        "id_cols": ["ip.src_host", "ip.dst_host", "tcp.srcport", "tcp.dstport", "udp.port"],
        "time_cols": ["frame.time"],
        "label_col": "Attack_label",
        "type_col": "Attack_type",
    },
}

summary_rows = []
label_dist_report = {}
type_dist_report = {}

for name, cfg in DATASETS.items():
    print(f"\n{'='*70}\n{name}\n{'='*70}")
    df = pd.read_csv(cfg["path"], low_memory=False)
    n_before = len(df)

    df = df.drop_duplicates()
    df = df.dropna(subset=[cfg["label_col"], cfg["type_col"]])
    print(f"Rows: {n_before} -> {len(df)} after dedup/label-dropna")

    reserved = set(cfg["id_cols"]) | set(cfg["time_cols"]) | {cfg["label_col"], cfg["type_col"]}
    feature_cols = [c for c in df.columns if c not in reserved]

    categorical_cols, dropped_cols, numeric_cols = [], [], []

    for c in feature_cols:
        is_textlike = (pd.api.types.is_object_dtype(df[c])
                       or pd.api.types.is_string_dtype(df[c]))
        if is_textlike:
            n_unique = df[c].replace("-", np.nan).nunique(dropna=True)
            if n_unique <= HIGH_CARDINALITY_THRESHOLD:
                categorical_cols.append(c)
            else:
                dropped_cols.append(c)
        else:
            numeric_cols.append(c)

    print(f"  Categorical (encoded): {len(categorical_cols)} -> {categorical_cols}")
    print(f"  Dropped (free-text/high-cardinality): {len(dropped_cols)} -> {dropped_cols}")
    print(f"  Numeric (normalized): {len(numeric_cols)}")

    df = df.drop(columns=dropped_cols)

    for c in categorical_cols:
        df[c] = df[c].replace("-", "unknown").astype(str)
        df[c] = df[c].astype("category").cat.codes

    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
        col_min, col_max = df[c].min(), df[c].max()
        if col_max > col_min:
            df[c] = (df[c] - col_min) / (col_max - col_min)
        else:
            df[c] = 0.0

    out_path = f"data/processed/{name.replace(' ', '_')}_processed.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved processed file -> {out_path}  shape={df.shape}")

    label_counts = df[cfg["label_col"]].value_counts().to_dict()
    type_counts = df[cfg["type_col"]].value_counts().to_dict()
    n_total = len(df)
    benign_pct = 100 * label_counts.get(0, 0) / n_total
    attack_pct = 100 - benign_pct

    print(f"  Binary label distribution: {label_counts}")
    print(f"  Attack-type distribution: {type_counts}")

    label_dist_report[name] = label_counts
    type_dist_report[name] = type_counts

    n_feat_final = len([c for c in df.columns if c not in
                         ({cfg['label_col'], cfg['type_col']} | set(cfg['id_cols']) | set(cfg['time_cols']))])

    summary_rows.append({
        "Dataset": name,
        "Records_after_cleaning": n_total,
        "Num_features_final": n_feat_final,
        "Num_attack_classes": df[cfg["type_col"]].nunique(),
        "Benign_pct": round(benign_pct, 3),
        "Attack_pct": round(attack_pct, 3),
    })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv("outputs/tables/table2_dataset_summary.csv", index=False)
print("\n\n=== FINAL TABLE 2 (dataset summary) ===")
print(summary_df.to_string(index=False))

print("\n\n=== FULL TYPE DISTRIBUTIONS (for manuscript detail) ===")
for name, d in type_dist_report.items():
    print(f"\n{name}:")
    total = sum(d.values())
    for k, v in sorted(d.items(), key=lambda x: -x[1]):
        print(f"  {k:<25} {v:>10}  ({100*v/total:.3f}%)")
