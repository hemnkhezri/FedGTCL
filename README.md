# FedGTCL: Federated Graph-Temporal Contrastive Learning for IIoT Intrusion Detection

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23592623.svg)](https://doi.org/10.5281/zenodo.23592623)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Code for the paper

> H. Khezri and M. Trik, *Reliable Federated Intrusion Detection for Industrial IoT under Non-IID Heterogeneity:
> A Lightweight Graph-Temporal Contrastive Learning Approach* (under review; journal and DOI to be added).

FedGTCL trains an intrusion detector across industrial sites without moving raw traffic off site. It has three parts:

1. **Lightweight graph-temporal encoder** (`src/04_model.py`, Section 3.3). Each 60-second traffic window
   becomes a host-level device-interaction graph (`src/03_graph_construction.py`). The encoder applies
   top-m sparsified, low-rank graph attention and a depthwise-separable temporal convolution over T=5 windows.
   It has 18,565 parameters, 42.2% fewer than a matched dense GCN-GRU (32,133).
2. **Structure-aware contrastive objective** (Section 3.4). An NT-Xent loss whose negatives are re-weighted by
   host-set overlap is combined with a supervised loss on the labeled fraction of each client's windows.
3. **FedAdaptOpt** (Section 3.5). Clients upload top-s sparsified updates (s = 0.3), and the server applies a
   momentum-corrected, Adam-style update.

## What the paper reports

The evaluation is a pilot at reduced scale (K=4 clients, R=10 rounds, E=3 local epochs) on WUSTL-IIoT-2021,
TON_IoT-Network and a stratified 10% subsample of Edge-IIoTset, under IID and Dirichlet non-IID (α=0.3) partitions.

- Under non-IID partitions with partial labels, FedAvg-GCN-GRU collapsed to a constant predictor in all 7
  WUSTL-IIoT-2021 partition/seed trials (FedAvg-LSTM in all 4 of its trials), and both FedAvg baselines
  collapsed on TON_IoT-Network and Edge-IIoTset.
- FedGTCL avoided that collapse in 3 of the 7 WUSTL-IIoT-2021 trials and stayed non-degenerate on TON_IoT-Network
  (F1 58.65%) and Edge-IIoTset (F1 88.17%).
- The advantage is probabilistic, not absolute. FedGTCL also collapsed in 4 of 7 WUSTL trials, and results vary
  strongly across seeds at this round budget (Table 12).
- On two further datasets run under the same protocol (Online Resource 1, Table S7), the FedAvg baselines again
  collapsed in all four federated settings, and FedGTCL avoided collapse in three of them without becoming a
  useful detector: on X-IIoTID its F1 stayed below the always-attack rule, and on CICAPT-IIoT2024 (2% attack
  windows) it found 2 of 15 attack sequences under IID and none under non-IID.

- **Scope of the claim.** A later pre-registered re-evaluation by the same authors (manuscript under review;
  code at https://github.com/hemnkhezri/fed-ids-evaluation) used window-disjoint chronological validation splits.
  At this budget (30 local steps per client) it confirmed the collapse pattern; with 1,000 local steps the baselines
  rarely collapsed and FedGTCL was not better than FedProx. The advantage reported here is specific to short local
  training (paper, Section 5).

## Repository layout

```
src/                         main pipeline, numbered in run order
  01_make_edgeiiot_subsample.py   rebuild the exact Edge-IIoTset 10% subsample
  02_preprocess.py                cleaning, encoding, normalization of all three datasets (Table 3)
  03_graph_construction.py        60-s windows -> host-level graphs (module)
  04_model.py                     FedGTCL encoder (module)
  05_test_encoder.py              parameter counts and latency (Table 2)
  06a_/06_federated_training.py   FedGTCL with FedAdaptOpt (Table 6)
  07_shared_setup.py              canonical seed-42 IID / non-IID partition
  07b/07c/07d_*.py                FedAvg-GCN-GRU, Centralized GCN-GRU, FedAvg-LSTM (Tables 6 and 8)
  08*, 09*                        TON_IoT-Network and Edge-IIoTset pipelines + baselines (Tables 9-10)
  10-12_*.py                      ablation, multi-seed, R=30 scale-up (Tables 11-13)
  13-15_*.py                      values behind Figs. 7, 13 and 12
  16*_*.py                        X-IIoTID and CICAPT-IIoT2024 (Online Resource 1, Table S7)
experiments/
  sensitivity/                partition seeds (Table 7), α (S1), K (S2), labeled fraction (S3)
  additional_baselines/       client-side error feedback (S5), FedProx and SCAFFOLD (S6)
analysis/                     parameter accounting, coordinate overlap, host-cap statistics,
                              synthetic |V| scaling (S4), partition/time-span checks
figures/
  scripts/                    fig01_iiot_architecture.py (Fig. 1) and make_figures.py (Figs. 2-13), all vector-drawn
  final/                      the 13 figures as they appear in the manuscript (PNG 600 dpi and vector PDF)
data/                         download instructions, checksums, Edge-IIoTset row indices
docs/REPRODUCIBILITY.md       every table and figure -> script -> command, and a verification run
results/logs/                 logs of that verification run
run_all.sh                    runs everything in paper order
```

Every script starts with a `# Paper output:` line naming the table, figure or section it produces.
Some scripts keep their working names from the study: `07c_baseline_centralized_gaco.py` is the
*Centralized GCN-GRU* baseline of the paper, and log headers that say "GACO" refer to it.

## Quick start

```bash
pip install -r requirements.txt          # Python 3.11, CPU only
# put the raw datasets in data/raw/ (see data/README.md), then:
bash run_all.sh core                     # data, partitions, Tables 2-3 and 6-13 (under an hour on one core)
bash run_all.sh                          # everything, including the Online Resource 1 tables (about 2 h)
bash run_all.sh figures                  # redraw the 13 figures only (seconds; no data needed)
```

`run_all.sh` sets `OMP_NUM_THREADS=1`. Keep that setting when running scripts on their own: the published values
are reproduced with single-threaded PyTorch.

All scripts are run from the repository root. They read `data/raw/`, write intermediate files to
`data/processed/`, and print their results to the console (`run_all.sh` also saves them in `logs/`).

## Reproducibility

`docs/REPRODUCIBILITY.md` lists the script for every table and figure, and compares the paper's values with a
complete re-run of the core pipeline from the raw files. In short:

- **Exact:** every accuracy, F1 and confusion-matrix value of Tables 3 and 6-13 and of Online Resource 1, Tables
  S1-S3 and S5-S7, the partitions and counts of Sections 4.1, 4.3 and 4.7, the parameter counts of Table 2,
  the numbers quoted in Sections 3.5, 3.6 and 4.10, and all 13 figures (pixel-identical).
- **Script version matters:** FedGTCL at R=10 is sensitive to the random-number state at the start of training
  (Section 4.12). The document names the script version behind each published FedGTCL value; for Table 6, the
  IID row comes from `06a_federated_training_ownsplit.py` and the non-IID row from `06_federated_training.py`.
- **Hardware-dependent:** wall-clock times (Table 2 latency, Fig. 13, Table S4). The direction of the results is
  stable; the absolute times and, to a lesser degree, the ratios are not.

## License and citation

Code: MIT (see `LICENSE`). The datasets keep their own licenses and are not redistributed here.
Please cite the paper (see `CITATION.cff`).

## Archive

This repository is archived on Zenodo: https://doi.org/10.5281/zenodo.23592623
