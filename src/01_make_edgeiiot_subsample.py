# Paper output: Section 4.1: rebuilds the exact Edge-IIoTset 10% subsample
# Run from the repository root.
"""
Step 1: rebuild the Edge-IIoTset 10% stratified subsample used in the paper (Section 4.1).

Input : data/raw/DNN-EdgeIIoT-dataset.csv  (official Edge-IIoTset release, 2,219,201 rows)
Output: data/raw/edgeiiot_sample2_200k.csv (221,917 rows; floor(10%) of every Attack_type class)

The subsample was drawn once, before any experiment. To make it exactly reproducible without
redistributing the dataset, this repository ships the 0-based row positions of the drawn rows
in the official file (data/edgeiiot_subsample/row_indices.txt.gz), in the order in which they
appear in the subsample. This script selects those rows and writes them unchanged.

Run from the repository root:  python src/01_make_edgeiiot_subsample.py
"""
import gzip
import numpy as np
import pandas as pd

SRC = "data/raw/DNN-EdgeIIoT-dataset.csv"
IDX = "data/edgeiiot_subsample/row_indices.txt.gz"
OUT = "data/raw/edgeiiot_sample2_200k.csv"

with gzip.open(IDX, "rt") as f:
    order = np.array([int(x) for x in f.read().split()], dtype=np.int64)
wanted = set(order.tolist())
print(f"Selecting {len(order):,} rows from {SRC}")

parts, base = [], 0
for chunk in pd.read_csv(SRC, dtype=str, keep_default_na=False, chunksize=200_000, low_memory=False):
    chunk.index = np.arange(base, base + len(chunk))
    parts.append(chunk[chunk.index.isin(wanted)])
    base += len(chunk)
pool = pd.concat(parts)
missing = wanted - set(pool.index.tolist())
assert not missing, f"{len(missing)} row positions not found; is {SRC} the official 2,219,201-row file?"
sample = pool.loc[order]
sample.to_csv(OUT, index=False)
print(f"Wrote {len(sample):,} rows to {OUT}")
print(sample["Attack_type"].value_counts().to_string())
