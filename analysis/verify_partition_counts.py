# Paper output: Sections 4.1 and 4.7 (window counts, WUSTL time span, per-client attack counts)
# Run from the repository root.
# Runs the unmodified shared setup, then reports the counts quoted in Sections 4.1, 4.3 and 4.7.
import runpy, pickle, numpy as np, pandas as pd
g = runpy.run_path("src/07_shared_setup.py")
seqs = g["all_sequences"]; W = g["windows_data"]
lab = np.array([s["label"] for s in seqs])
print("\n=== Windows, sequences and per-client counts (Sections 4.3, 4.7; Table 7 seed 42) ===")
print("windows", len(W), "attack windows", sum(w["label"] for w in W))
print("sequences", len(seqs), "attack sequences", lab.sum())
for name in ("iid", "noniid"):
    tr, va = g[name + "_train"], g[name + "_val"]
    print(name, "shard sizes", [len(a) + len(b) for a, b in zip(tr, va)],
          "shard attack", [sum(s["label"] for s in a) + sum(s["label"] for s in b) for a, b in zip(tr, va)])
    print(name, "train sizes", [len(a) for a in tr], "train attack", [sum(s["label"] for s in a) for a in tr])
    print(name, "val sizes", [len(b) for b in va], "val attack", [sum(s["label"] for s in b) for b in va],
          "total val", sum(len(b) for b in va), "val attack total", sum(sum(s["label"] for s in b) for b in va))
print("\n=== WUSTL-IIoT-2021 time span (Section 4.1) ===")
df = pd.read_csv("data/processed/WUSTL-IIoT-2021_processed.csv", usecols=["StartTime"])
t = pd.to_datetime(df["StartTime"]).sort_values()
print("first", t.iloc[0], "last", t.iloc[-1], "span h", (t.iloc[-1] - t.iloc[0]).total_seconds() / 3600)
wid = ((t - t.iloc[0]).dt.total_seconds() // 60).astype(int)
print("possible 60s buckets in span", wid.max() + 1, "non-empty buckets", wid.nunique())
u = np.sort(wid.unique()); gaps = np.diff(u); big = np.where(gaps > 1)[0]
print("number of gaps", len(big), "largest gaps (min)", sorted(gaps[big])[-5:] if len(big) else [])
print("distinct dates", t.dt.date.value_counts().sort_index().to_dict())
