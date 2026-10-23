# Reproducibility

This page maps every table and figure of the paper to the script that produces it, and reports a complete
re-run of the pipeline from the raw dataset files (Python 3.11, PyTorch 2.14.0 CPU, 2-core Linux machine,
`OMP_NUM_THREADS=1`). The logs of that run are in `results/logs/`.

## How to read the status column

- **exact**: the re-run printed the same numbers as the paper (to the reported precision, including the
  confusion-matrix counts where the paper gives them).
- **hardware**: wall-clock timings. They depend on the CPU and system load; see the notes below.

## Two settings that matter

1. **Single-threaded PyTorch.** With the default multi-threaded CPU kernels, floating-point sums are taken in a
   different order and some FedGTCL runs end differently. Example: TON_IoT-Network non-IID gives F1 58.65%
   (the published value) with one thread and 87.20% (an always-attack collapse) with two. `run_all.sh` sets
   `OMP_NUM_THREADS=1`; set it yourself when running scripts individually.
2. **Random-number state.** FedGTCL at R=10 is sensitive to the random-number state at the start of training
   (Section 4.12 of the paper quantifies this across seeds). Several published values therefore depend on the
   exact script version, not only on the seed. The table below names the version for each value:
   - `src/06a_federated_training_ownsplit.py` is the earlier version of the FedGTCL training script. It draws
     the partition itself, and its IID train/validation split is identical to the shared partition of
     `src/07_shared_setup.py`. It produced the **FedGTCL IID row of Table 6**. Its non-IID split differs (23
     instead of 27 validation attack sequences), so it is not used for any non-IID result.
   - `src/06_federated_training.py` trains on the shared partition. With seed 42 it reproduces the **FedGTCL
     non-IID row of Table 6**. (The same seed gives IID F1 57.53%, not 73.47%, because the random-number state
     at the start of training differs from 06a's.)
   - The scripts derived from it for Table 7 and Online Resource 1, Tables S1, S2 and S5
     (`experiments/sensitivity/06_federated_training_{param,alpha,k}.py`,
     `experiments/additional_baselines/06_federated_training_ef*.py`) set `SEED = 123` for training, as run.
   - `analysis/compute_coord_overlap.py` executes the header of `06_federated_training.py`; it switches that
     header to `SEED = 123`, the value in force when the published overlap figures were computed.

## Training details as implemented

Two details of the implementation matter when reading the results (both are stated in Sections 4.4 and 4.7 of
the paper):

- A "local epoch" is one Adam step on a mini-batch of up to 16 training sequences drawn at random from the
  client's shard. With E=3 and R=10, each client takes 3 steps per round and 30 in total. The centralized
  baseline (`07c`) instead makes 40 full passes over its training set in mini-batches of 16 (about 840 steps).
- The FedAvg, FedProx and SCAFFOLD baselines use the labels of all of a client's training windows. The labeled
  fraction p_L=0.5 applies only to FedGTCL and its ablations (`labeled_anchor_idxs` in the FedGTCL scripts).

## Main tables

| Paper | Script(s) | Command | Status |
|---|---|---|---|
| Table 2, parameter counts | `src/05_test_encoder.py` | `python src/05_test_encoder.py` | exact (18,565 vs 32,133; 42.2%) |
| Table 2, latency | same | same | hardware (paper 0.886 / 0.396 ms per T=5 sequence, ratio 2.24; unloaded re-run 1.45-1.50 / 0.65-0.66 ms, ratio about 2.2; under load the ratio drifts) |
| Table 3, Section 4.1 | `src/02_preprocess.py` | `python src/02_preprocess.py` | exact |
| Section 4.1 (WUSTL time span), 4.7 (per-client counts) | `analysis/verify_partition_counts.py` | `python analysis/verify_partition_counts.py` | exact (423 windows, 7.04 h; 0, 0, 112, 1) |
| Table 6, FedGTCL IID | `src/06a_federated_training_ownsplit.py` | `python src/06a_federated_training_ownsplit.py` | exact (84.52 / 73.47; TP 18, FP 9, TN 53, FN 4) |
| Table 6, FedGTCL non-IID | `src/06_federated_training.py` | `python src/06_federated_training.py` | exact (68.24 / 0.00; TN 58, FN 27) |
| Table 6, FedAvg-GCN-GRU | `src/07b_baseline_fedavg_gcngru.py` | `python src/07b_baseline_fedavg_gcngru.py` | exact |
| Table 6, FedAvg-LSTM | `src/07d_baseline_fedavg_lstm.py` | `python src/07d_baseline_fedavg_lstm.py` | exact |
| Table 7 | `experiments/sensitivity/07_shared_setup_param.py <seed>`, then `06_federated_training_param.py`, `07b_*_param.py`, `07d_*_param.py` with seed 100, 200, 300 | see `run_all.sh` | exact (FedGTCL 0.00, 57.63, 46.51; baselines 0.00) |
| Table 8 | `src/07c_baseline_centralized_gaco.py` | `python src/07c_baseline_centralized_gaco.py` | exact (97.62 / 94.44; TP 17, FP 0) |
| Table 9 | `src/08_ton_iot_full_pipeline.py`, `src/08b_ton_iot_baselines.py` | `python src/08_...py && python src/08b_...py` | exact (single-threaded) |
| Table 10 | `src/09_edgeiiot_full_pipeline.py`, `src/09b_edgeiiot_baselines.py` | `python src/09_...py && python src/09b_...py` | exact |
| Table 11 | `src/10_ablation_study.py` | `python src/10_ablation_study.py` | exact (all six rows and TN values) |
| Table 12 | `src/11_multiseed_robustness.py` (needs partitions 42, 123, 2024 from `07_shared_setup_param.py`) | `python src/11_multiseed_robustness.py` | exact per-seed values (the script prints the population SD; the paper reports the sample SD, n=3) |
| Table 13 | `src/12_scaleup_convergence.py` | `python src/12_scaleup_convergence.py` | exact (all 12 checkpoints) |

Each seed of Table 12 draws its own partition with the procedure of Section 4.3 (`shared_partition_<seed>.pkl`);
seed 42 is the partition used by Tables 6, 11 and 13.

## Online Resource 1 tables

| Paper | Script(s) | Status |
|---|---|---|
| Table S1 (α) | `experiments/sensitivity/07_shared_setup_alpha.py 42 <α>`, then `06_/07b_/07d_*_alpha.py 42 <α>` | exact (non-IID F1: FedGTCL 46.67, 0.00, 53.33, 0.00; FedAvg-GCN-GRU 0.00 at every α; FedAvg-LSTM 57.14 at α=0.5, else 0.00) |
| Table S2 (K) | `experiments/sensitivity/07_shared_setup_k.py 42 0.3 <K>`, then `06_/07b_/07d_*_k.py 42 0.3 <K>` | exact (K=8: FedGTCL 33.33, both baselines 0.00 with TN=65; K=10: FedGTCL 48.89, both baselines 43.24 with TN=0) |
| Table S3 (p_L) | `experiments/sensitivity/10_ablation_study_pL_sweep.py` | exact (including TN=58 for the full model at p_L=0.20) |
| Table S4 (synthetic \|V\|) | `analysis/synthetic_scaling.py` | hardware (paper: attention 1,440x, dense 4.7x from \|V\|=10 to 800; re-run 2,643x and 8.9x; attention grows super-linearly in both) |
| Table S5 (error feedback) | `experiments/additional_baselines/06_federated_training_ef.py` (seed 42), `..._ef_seed2024.py` | exact for all four 'with EF' values (seed 42: 0.00 IID, 48.21 non-IID; seed 2024: 48.28, 50.98) and for the seed-2024 'without EF' values (42.42, 50.98, from `experiments/sensitivity/06_federated_training_param.py 2024`). The seed-42 'without EF' values (72.34 IID, 0.00 non-IID) are the seed-42 results of `src/10_ablation_study.py` / `src/11_multiseed_robustness.py` (Tables 11-12); `06_federated_training_param.py 42` gives 8.70 / 0.00. |
| Table S6 (FedProx, SCAFFOLD) | `experiments/additional_baselines/07e_baseline_fedprox_gcngru.py` (μ=0.1), `..._mu001.py` (μ=0.01), `07f_baseline_scaffold_gcngru.py` | exact |
| Table S7 (X-IIoTID, CICAPT-IIoT2024) | `src/16a_prepare_xiiotid.py`, `src/16a_prepare_cicapt.py`, then `src/16_extra_dataset_pipeline.py <xiiotid\|cicapt>` and `src/16b_extra_dataset_baselines.py <xiiotid\|cicapt>` | values as run for the table (single-threaded, seed 42; logs `16*_xiiotid.log`, `16*_cicapt.log`) |

## Other numbers quoted in the text

| Paper | Script | Status |
|---|---|---|
| Section 3.5, Jaccard overlap of top-s coordinates | `analysis/compute_coord_overlap.py` | exact (IID 0.31, std 0.09; non-IID 0.26, std 0.24) |
| Section 3.6, parameter accounting | `analysis/decompose_params.py` | exact (GRU 24,960; low-rank saving 3,360; measured saving 13,568) |
| Section 4.10, Edge-IIoTset host cap | `analysis/compute_hostcap_stats.py` | exact (18 of 796 windows cap-binding; 12,193 unparseable rows, 5.50%) |

## Figures

`figures/scripts/make_figures.py` draws Figs. 2-13 and `figures/scripts/fig01_iiot_architecture.py` draws Fig. 1.
They plot the values reported in the paper (copied from the tables and from the logs of `src/13_*.py`, `src/14_*.py`,
`src/15_*.py`), so they run in seconds without the data: `bash run_all.sh figures`. The output is pixel-identical to
the images embedded in the manuscript. Fig. 1 is drawn as vector art by `fig01_iiot_architecture.py`. Figs. 2-13
are drawn at their printed width and saved as 600-dpi PNG and vector PDF.

| Figure | Function in `make_figures.py` | Values from |
|---|---|---|
| 1 | `fig01_iiot_architecture.py` | diagram |
| 2 | `fig2` | diagram (method architecture) |
| 3 | `fig3` | diagram (federated topology) |
| 4 | `fig4` | diagram (graph construction) |
| 5 | `fig5` | diagram (encoder) |
| 6 | `fig6` | diagram (Algorithm 1, per-round flow) |
| 7 | `fig8` | `src/13_cost_curve_data.py` (exact) |
| 8 | `fig10` | Tables 6, 9, 10 |
| 9 | `fig11` | Table 11 |
| 10 | `fig12` | Table 12 |
| 11 | `fig14` | Table 13 |
| 12 | `fig15` | `src/15_multiclass_per_attack_accuracy.py` (exact) |
| 13 | `fig16` | `src/14_scalability_analysis.py` (hardware) |

## Timing measurements

Table 2, Fig. 13 and Table S4 are wall-clock times on one CPU core. Absolute values change with the processor,
PyTorch build and background load, and so, to a smaller degree, do the ratios between methods. What is stable is
the direction: the FedGTCL encoder is slower than the dense baseline at the host counts of these datasets, and
the attention layer's cost grows faster than linearly with |V| under the current dense-masked implementation.
Fig. 13 times only the encoder forward passes over all T=5 sequences (sequential, unbatched); graph
construction is excluded from the timed region.
