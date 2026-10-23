# Paper output: Online Resource 1, Table S7 (window graphs for X-IIoTID)
# Run from the repository root.
"""
X-IIoTID window graphs for Online Resource 1, Table S7 (same windowing rules as cicapt_prepare.py).

Input : data/raw/X-IIoTID_dataset.csv (820,834 rows, 68 columns)
Output: data/processed/xiiotid_windows.pkl

Rules: label = (class3 == "Attack"); 60-s windows on the Unix Timestamp; rows with an unparseable Timestamp or a
source/destination that is not a dotted IPv4 address are dropped; Date, Timestamp, Scr_IP, Des_IP, Scr_port,
Des_port and class1-3 are excluded from the features; Protocol and Service are integer-encoded; booleans become
0/1 and other non-numeric tokens 0. Min-max normalisation uses the rows of windows in chronological blocks 1, 3
and 5 of five equal blocks. Each window keeps its 30 most active hosts; node features are the mean normalised
features of the flows touching the host; edges keep each host's top-8 partners by flow count; a window is labelled
attack if any of its flows is an attack.
Run from the repository root:  python src/16a_prepare_xiiotid.py
"""
import pickle, hashlib, re
import numpy as np
import pandas as pd

SRC, OUT = "data/raw/X-IIoTID_dataset.csv", "data/processed/xiiotid_windows.pkl"
TOP_M, MAX_NODES, WIN = 8, 30, 60
EXCL = ["Date", "Timestamp", "Scr_IP", "Des_IP", "Scr_port", "Des_port", "class1", "class2", "class3"]

df = pd.read_csv(SRC, low_memory=False, dtype=str, keep_default_na=False)
n0 = len(df)
df["ts"] = pd.to_numeric(df["Timestamp"], errors="coerce")
ip = re.compile(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$")
ok = df["ts"].notna() & df["Scr_IP"].str.match(ip) & df["Des_IP"].str.match(ip)
df = df[ok].reset_index(drop=True)
print(f"rows kept {len(df):,} of {n0:,}")
FEATS = [c for c in df.columns if c not in EXCL + ["ts"]]
X = pd.DataFrame(index=df.index)
for c in FEATS:
    if c in ("Protocol", "Service"):
        X[c] = df[c].astype("category").cat.codes.astype(float)
    else:
        v = df[c].replace({"True": "1", "False": "0", "true": "1", "false": "0"})
        X[c] = pd.to_numeric(v, errors="coerce").fillna(0.0)
y = (df["class3"] == "Attack").astype(int).to_numpy()
tmin = df["ts"].min()
b = ((df["ts"] - tmin) // WIN).astype(np.int64).to_numpy()
buckets = np.sort(np.unique(b)); W = len(buckets); widx = {k: i for i, k in enumerate(buckets)}
bounds = [int(round(i * W / 5)) for i in range(6)]
region = np.empty(W, dtype=object); segment = np.zeros(W, dtype=int)
for bl in range(5):
    region[bounds[bl]:bounds[bl + 1]] = "val" if bl in (1, 3) else "train"
    segment[bounds[bl]:bounds[bl + 1]] = bl
wi = np.array([widx[k] for k in b])
train_rows = np.array([region[i] == "train" for i in wi])
lo = X[train_rows].min(); hi = X[train_rows].max(); rng = (hi - lo).replace(0, np.nan)
Xn = ((X - lo) / rng).fillna(0).clip(0, 1).to_numpy(np.float64)
src = df["Scr_IP"].to_numpy(); dst = df["Des_IP"].to_numpy()

order = np.argsort(wi, kind="stable"); starts = np.searchsorted(wi[order], np.arange(W + 1))
windows = []
for i in range(W):
    rows = order[starts[i]:starts[i + 1]]
    cnt = {}
    for h in np.concatenate([src[rows], dst[rows]]):
        cnt[h] = cnt.get(h, 0) + 1
    keep = set(h for h, _ in sorted(cnt.items(), key=lambda x: (-x[1], x[0]))[:MAX_NODES])
    rows_k = [r for r in rows if src[r] in keep and dst[r] in keep]
    hosts = sorted(set(src[rows_k]) | set(dst[rows_k]))
    if not hosts:
        windows.append(None); continue
    ix = {h: j for j, h in enumerate(hosts)}; N = len(hosts)
    fs = np.zeros((N, Xn.shape[1])); nc = np.zeros(N)
    ew = np.zeros((N, N), np.float32)
    for r in rows_k:
        s, t = ix[src[r]], ix[dst[r]]
        fs[s] += Xn[r]; nc[s] += 1; fs[t] += Xn[r]; nc[t] += 1
        ew[s, t] += 1; ew[t, s] += 1
    feats = (fs / nc[:, None]).astype(np.float32)
    adj = np.zeros((N, N), bool)
    for v in range(N):
        row = ew[v].copy(); row[v] = -1
        me = min(TOP_M, int((row > 0).sum()))
        if me > 0:
            top = np.argpartition(-row, me - 1)[:me]; adj[v, top[row[top] > 0]] = True
    windows.append(dict(win=i, node_ids=hosts, feats=feats, mask=adj, label=int(y[rows].max()),
                        n_flows=len(rows_k)))
out = dict(dataset="xiiotid", feature_cols=FEATS, windows=windows, region=region.tolist(),
           segment=segment.tolist(), tmin=float(tmin), W=W)
pickle.dump(out, open(OUT, "wb"))
print(f"windows {W}, attack windows {sum(w['label'] for w in windows if w)}, empty {sum(w is None for w in windows)}")
print("sha256", hashlib.sha256(open(OUT, "rb").read()).hexdigest())
