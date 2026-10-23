#!/usr/bin/env bash
# Runs the complete FedGTCL pipeline in paper order. Run from the repository root after placing the raw
# datasets in data/raw/ (see data/README.md). Every script prints its results; this wrapper keeps a copy
# of each output in logs/<script>.log. CPU only, single-threaded; the full run takes several hours.
#
#   bash run_all.sh            # everything
#   bash run_all.sh core       # data, partitions and the main tables (2, 3, 5-11)
#   bash run_all.sh figures    # redraw all 13 figures from the values reported in the paper
set -euo pipefail
# The published values are reproduced with single-threaded PyTorch; multi-threaded CPU kernels change the
# floating-point summation order and, through it, some FedGTCL results (docs/REPRODUCIBILITY.md).
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
mkdir -p logs data/processed figures/final
STAGE="${1:-all}"

run() {  # run <script> [args...]
  local name; name="$(basename "$1" .py)${2:+_$2}${3:+_$3}${4:+_$4}"
  echo ">>> $*"
  python "$@" 2>&1 | tee "logs/${name}.log"
}

if [[ "$STAGE" == "all" || "$STAGE" == "core" ]]; then
  # --- data (Section 4.1, Table 3) ---
  [[ -f data/raw/edgeiiot_sample2_200k.csv ]] || run src/01_make_edgeiiot_subsample.py
  run src/02_preprocess.py
  # --- partitions (Section 4.3) ---
  run src/07_shared_setup.py                                   # canonical seed-42 partition
  for s in 42 123 2024 100 200 300; do                         # Tables 7 and 12
    run experiments/sensitivity/07_shared_setup_param.py "$s"
  done
  run analysis/verify_partition_counts.py                      # Sections 4.1 and 4.7 counts
  # --- WUSTL-IIoT-2021 (Tables 2, 6, 8) ---
  run src/05_test_encoder.py
  run src/06a_federated_training_ownsplit.py                  # Table 6, FedGTCL IID row
  run src/06_federated_training.py                             # Table 6, FedGTCL non-IID row
  run src/07b_baseline_fedavg_gcngru.py
  run src/07c_baseline_centralized_gaco.py
  run src/07d_baseline_fedavg_lstm.py
  # --- TON_IoT-Network (Table 9) and Edge-IIoTset (Table 10) ---
  run src/08_ton_iot_full_pipeline.py
  run src/08b_ton_iot_baselines.py
  run src/09_edgeiiot_full_pipeline.py
  run src/09b_edgeiiot_baselines.py
  # --- ablation, multi-seed, scale-up (Tables 11-13) ---
  run src/10_ablation_study.py
  run src/11_multiseed_robustness.py
  run src/12_scaleup_convergence.py
fi

if [[ "$STAGE" == "all" ]]; then
  # --- values behind Figs. 7, 12, 13 ---
  run src/13_cost_curve_data.py
  run src/14_scalability_analysis.py
  run src/15_multiclass_per_attack_accuracy.py
  # --- Table 7: four partition seeds ---
  for s in 100 200 300; do
    run experiments/sensitivity/06_federated_training_param.py "$s"
    run experiments/sensitivity/07b_baseline_fedavg_gcngru_param.py "$s"
    run experiments/sensitivity/07d_baseline_fedavg_lstm_param.py "$s"
  done
  # --- Online Resource 1, Table S1: Dirichlet alpha ---
  for a in 0.1 0.3 0.5 1.0; do
    run experiments/sensitivity/07_shared_setup_alpha.py 42 "$a"
    run experiments/sensitivity/06_federated_training_alpha.py 42 "$a"
    run experiments/sensitivity/07b_baseline_fedavg_gcngru_alpha.py 42 "$a"
    run experiments/sensitivity/07d_baseline_fedavg_lstm_alpha.py 42 "$a"
  done
  # --- Online Resource 1, Table S2: client count K ---
  for k in 8 10; do
    run experiments/sensitivity/07_shared_setup_k.py 42 0.3 "$k"
    run experiments/sensitivity/06_federated_training_k.py 42 0.3 "$k"
    run experiments/sensitivity/07b_baseline_fedavg_gcngru_k.py 42 0.3 "$k"
    run experiments/sensitivity/07d_baseline_fedavg_lstm_k.py 42 0.3 "$k"
  done
  # --- Online Resource 1, Tables S3-S6 ---
  run experiments/sensitivity/10_ablation_study_pL_sweep.py
  run analysis/synthetic_scaling.py
  run experiments/additional_baselines/06_federated_training_ef.py
  run experiments/additional_baselines/06_federated_training_ef_seed2024.py
  run experiments/additional_baselines/07e_baseline_fedprox_gcngru.py
  run experiments/additional_baselines/07e_baseline_fedprox_gcngru_mu001.py
  run experiments/additional_baselines/07f_baseline_scaffold_gcngru.py
  # --- Online Resource 1, Table S7 (needs data/raw/X-IIoTID_dataset.csv and data/raw/phase2_NetworkData.csv) ---
  if [[ -f data/raw/X-IIoTID_dataset.csv ]]; then
    run src/16a_prepare_xiiotid.py
    run src/16_extra_dataset_pipeline.py xiiotid
    run src/16b_extra_dataset_baselines.py xiiotid
  fi
  if [[ -f data/raw/phase2_NetworkData.csv ]]; then
    run src/16a_prepare_cicapt.py
    run src/16_extra_dataset_pipeline.py cicapt
    run src/16b_extra_dataset_baselines.py cicapt
  fi
  # --- numbers quoted in Sections 3.5, 3.6 and 4.10 ---
  run analysis/decompose_params.py
  run analysis/compute_coord_overlap.py
  run analysis/compute_hostcap_stats.py
fi

if [[ "$STAGE" == "all" || "$STAGE" == "figures" ]]; then
  run figures/scripts/fig01_iiot_architecture.py
  run figures/scripts/make_figures.py
fi
echo "Done. Outputs: logs/*.log and figures/final/fig*.png / fig*.pdf"
