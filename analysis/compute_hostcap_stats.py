# Paper output: Section 4.10 (Edge-IIoTset host-cap statistics)
# Run from the repository root.
import re
import numpy as np
import pandas as pd

df = pd.read_csv("data/processed/EdgeIIoTset-10pct_processed.csv", low_memory=False)

def parse_secs(s):
    m = re.search(r"(\d{2}):(\d{2}):(\d{2}\.\d+)", str(s))
    if not m:
        return None
    h, mi, se = m.groups()
    return int(h) * 3600 + int(mi) * 60 + float(se)

df["t_secs"] = df["frame.time"].apply(parse_secs)
n_before = len(df)
df = df.dropna(subset=["t_secs"]).sort_values("t_secs").reset_index(drop=True)
print(f"Dropped {n_before - len(df)} rows with unparseable frame.time ({(n_before-len(df))/n_before*100:.2f}%)")

df["window_id"] = ((df["t_secs"] - df["t_secs"].min()) // 60).astype(int)

MAX_NODES = 30

host_counts_per_window = []
window_has_ddos_udp = []
window_ddos_udp_hostcount = []

for w, wdf in df.groupby("window_id"):
    hosts = pd.unique(pd.concat([wdf["ip.src_host"], wdf["ip.dst_host"]]))
    n_hosts = len(hosts)
    host_counts_per_window.append(n_hosts)
    has_ddos = (wdf["Attack_type"] == "DDoS_UDP").any()
    window_has_ddos_udp.append(has_ddos)
    if has_ddos:
        window_ddos_udp_hostcount.append(n_hosts)

host_counts_per_window = np.array(host_counts_per_window)
window_ddos_udp_hostcount = np.array(window_ddos_udp_hostcount)

n_windows = len(host_counts_per_window)
n_binding = (host_counts_per_window > MAX_NODES).sum()
print(f"\nTotal windows: {n_windows}")
print(f"Host count per window: median={np.median(host_counts_per_window):.0f}, "
      f"mean={host_counts_per_window.mean():.1f}, "
      f"99th pct={np.percentile(host_counts_per_window,99):.0f}, "
      f"max={host_counts_per_window.max()}")
print(f"Windows where cap is binding (host count > {MAX_NODES}): {n_binding} "
      f"({n_binding/n_windows*100:.2f}%)")

n_ddos_windows = len(window_ddos_udp_hostcount)
n_ddos_binding = (window_ddos_udp_hostcount > MAX_NODES).sum()
print(f"\nWindows containing >=1 DDoS_UDP record: {n_ddos_windows}")
if n_ddos_windows > 0:
    print(f"  of these, cap-binding (host count > {MAX_NODES}): {n_ddos_binding} "
          f"({n_ddos_binding/n_ddos_windows*100:.2f}%)")
    print(f"  DDoS_UDP-window host-count stats: median={np.median(window_ddos_udp_hostcount):.0f}, "
          f"max={window_ddos_udp_hostcount.max()}")

# also check DDoS_TCP, DDoS_ICMP, DDoS_HTTP for completeness
for atype in ["DDoS_TCP", "DDoS_ICMP", "DDoS_HTTP", "Port_Scanning"]:
    counts = []
    for w, wdf in df.groupby("window_id"):
        if (wdf["Attack_type"] == atype).any():
            hosts = pd.unique(pd.concat([wdf["ip.src_host"], wdf["ip.dst_host"]]))
            counts.append(len(hosts))
    counts = np.array(counts)
    if len(counts) > 0:
        binding = (counts > MAX_NODES).sum()
        print(f"{atype}: {len(counts)} windows, {binding} cap-binding ({binding/len(counts)*100:.1f}%), "
              f"median hosts={np.median(counts):.0f}")
