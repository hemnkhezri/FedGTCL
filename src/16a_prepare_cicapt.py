# Paper output: Online Resource 1, Table S7 (window graphs for CICAPT-IIoT2024)
# Run from the repository root.
"""CICAPT-IIoT2024 preprocessing (CICAPT_PROTOCOL.md). Run locally; streams the CSV in chunks.

Online Resource 1, Table S7: window graphs for CICAPT-IIoT2024 (phase-2 network data). Same windowing rules as
16a_prepare_xiiotid.py: 60-s windows, 30 most active hosts per window, top-8 neighbours, min-max bounds from
chronological blocks 1, 3 and 5 of five, window labelled attack if any flow is an attack. Streams the CSV in chunks.

Usage (from the repository root):
    python src/16a_prepare_cicapt.py [data/raw/phase2_NetworkData.csv]
Output: data/processed/cicapt_windows.pkl and data/processed/cicapt_prepare_log.txt.
"""
import sys, time, pickle, hashlib
import numpy as np
import pandas as pd

PATH = sys.argv[1] if len(sys.argv) > 1 else "data/raw/phase2_NetworkData.csv"
CHUNK = 500_000
TOP_M, MAX_NODES, WIN = 8, 30, 60
RESERVED = ["ts", "Source IP", "Destination IP", "Source Port", "Destination Port", "Protocol_name",
            "label", "subLabel", "subLabelCat"]
ABS_TIME = ["max_duration", "min_duration", "sum_duration", "average_duration", "flow_idle_time", "IAT"]
log = open("data/processed/cicapt_prepare_log.txt", "w", encoding="utf-8")


def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); log.write(s + "\n"); log.flush()


cols = list(pd.read_csv(PATH, nrows=1).columns)
FEATS = [c for c in cols if c not in RESERVED and c not in ABS_TIME]
say("features:", len(FEATS)); say(FEATS)
t0 = time.time()

# ---------------- pass 1: time origin, non-empty windows, per-window host activity and labels
tmin = None
for ch in pd.read_csv(PATH, chunksize=CHUNK, usecols=["ts"]):
    m = ch["ts"].min(); tmin = m if tmin is None else min(tmin, m)
say("tmin", tmin)
bucket_label, host_cnt = {}, {}
for ch in pd.read_csv(PATH, chunksize=CHUNK, usecols=["ts", "Source IP", "Destination IP", "label"],
                      dtype={"Source IP": str, "Destination IP": str}):
    b = ((ch["ts"] - tmin) // WIN).astype(np.int64)
    for k, v in ch.groupby(b)["label"].max().items():
        bucket_label[k] = max(bucket_label.get(k, 0), int(v))
    for col in ("Source IP", "Destination IP"):
        for (k, h), v in ch.groupby([b, ch[col]]).size().items():
            host_cnt.setdefault(k, {}); host_cnt[k][h] = host_cnt[k].get(h, 0) + int(v)
    say(f"pass1 {time.time() - t0:.0f}s")
buckets = sorted(bucket_label)
W = len(buckets); widx = {k: i for i, k in enumerate(buckets)}
bounds = [int(round(i * W / 5)) for i in range(6)]
region = np.empty(W, dtype=object); segment = np.zeros(W, dtype=int)
for bl in range(5):
    region[bounds[bl]:bounds[bl + 1]] = "val" if bl in (1, 3) else "train"
    segment[bounds[bl]:bounds[bl + 1]] = bl
keep = {}
for k, d in host_cnt.items():
    hs = sorted(d.items(), key=lambda x: (-x[1], x[0]))
    keep[k] = set(h for h, _ in hs[:MAX_NODES])
say("windows", W, "attack windows", sum(bucket_label.values()))

# ---------------- pass 2: min/max over training-region rows
lo = pd.Series(np.inf, index=FEATS); hi = pd.Series(-np.inf, index=FEATS)
train_buckets = set(buckets[i] for i in range(W) if region[i] == "train")
for ch in pd.read_csv(PATH, chunksize=CHUNK, usecols=["ts"] + FEATS):
    b = ((ch["ts"] - tmin) // WIN).astype(np.int64)
    sub = ch.loc[b.isin(train_buckets), FEATS].apply(pd.to_numeric, errors="coerce").fillna(0)
    if len(sub):
        lo = np.minimum(lo, sub.min()); hi = np.maximum(hi, sub.max())
    say(f"pass2 {time.time() - t0:.0f}s")
rng = (hi - lo).replace(0, np.nan)

# ---------------- pass 3: aggregate node features and edges from rows between kept hosts
node_sum, node_cnt, edge_w = {}, {}, {}
for ch in pd.read_csv(PATH, chunksize=CHUNK, usecols=["ts", "Source IP", "Destination IP"] + FEATS,
                      dtype={"Source IP": str, "Destination IP": str}):
    b = ((ch["ts"] - tmin) // WIN).astype(np.int64).to_numpy()
    s = ch["Source IP"].to_numpy(); d = ch["Destination IP"].to_numpy()
    ok = np.fromiter((si in keep[bi] and di in keep[bi] for bi, si, di in zip(b, s, d)), bool, len(b))
    X = ((ch[FEATS].apply(pd.to_numeric, errors="coerce").fillna(0) - lo) / rng).fillna(0).clip(0, 1)
    X = X.to_numpy(np.float64)[ok]; b = b[ok]; s = s[ok]; d = d[ok]
    for ends in (s, d):
        key = pd.MultiIndex.from_arrays([b, ends])
        g = pd.DataFrame(X).groupby(key)
        sums = g.sum(); cnts = g.size()
        for kk, row in zip(sums.index, sums.to_numpy()):
            if kk in node_sum:
                node_sum[kk] += row
            else:
                node_sum[kk] = row.copy()
        for kk, c in cnts.items():
            node_cnt[kk] = node_cnt.get(kk, 0) + int(c)
    for (bi, si, di), c in pd.Series(1, index=pd.MultiIndex.from_arrays([b, s, d])).groupby(level=[0, 1, 2]).size().items():
        edge_w[(bi, si, di)] = edge_w.get((bi, si, di), 0) + int(c)
    say(f"pass3 {time.time() - t0:.0f}s")

# ---------------- build window graphs
by_win_nodes, by_win_edges = {}, {}
for (bi, h) in node_cnt:
    by_win_nodes.setdefault(bi, []).append(h)
for (bi, si, di), c in edge_w.items():
    by_win_edges.setdefault(bi, []).append((si, di, c))
windows = []
for i, bk in enumerate(buckets):
    hosts = sorted(by_win_nodes.get(bk, []))
    if not hosts:
        windows.append(None); continue
    ix = {h: j for j, h in enumerate(hosts)}; N = len(hosts)
    feats = np.stack([node_sum[(bk, h)] / node_cnt[(bk, h)] for h in hosts]).astype(np.float32)
    ew = np.zeros((N, N), np.float32)
    for si, di, c in by_win_edges.get(bk, []):
        ew[ix[si], ix[di]] += c; ew[ix[di], ix[si]] += c
    adj = np.zeros((N, N), bool)
    for v in range(N):
        row = ew[v].copy(); row[v] = -1
        me = min(TOP_M, int((row > 0).sum()))
        if me > 0:
            top = np.argpartition(-row, me - 1)[:me]; adj[v, top[row[top] > 0]] = True
    windows.append(dict(win=i, node_ids=hosts, feats=feats, mask=adj, label=int(bucket_label[bk]),
                        n_flows=int(sum(node_cnt[(bk, h)] for h in hosts) // 2)))
out = dict(dataset="cicapt", feature_cols=FEATS, windows=windows, region=region.tolist(), segment=segment.tolist(),
           tmin=float(tmin), W=W)
with open("data/processed/cicapt_windows.pkl", "wb") as f:
    pickle.dump(out, f)
say("empty windows after capping:", sum(w is None for w in windows))
say("sha256", hashlib.sha256(open("data/processed/cicapt_windows.pkl", "rb").read()).hexdigest())
say(f"done in {time.time() - t0:.0f}s")
