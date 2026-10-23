# Data

The datasets are not redistributed here. Download them from their official sources and place the files in
`data/raw/` under the names below. All scripts read from `data/raw/` and write to `data/processed/`.

| Dataset | Official source | File expected in `data/raw/` | Rows |
|---|---|---|---|
| WUSTL-IIoT-2021 | Washington University in St. Louis, IIoT dataset page (Zolanvari et al.) | `wustl_iiot_2021.csv` | 1,194,464 |
| TON_IoT (network, train/test subset) | UNSW Canberra TON_IoT repository, `Train_Test_datasets/Train_Test_Network_dataset/train_test_network.csv` | `ton_iot_network.csv` | 211,043 |
| Edge-IIoTset | Ferrag et al., IEEE DataPort / Kaggle, `DNN-EdgeIIoT-dataset.csv` | `DNN-EdgeIIoT-dataset.csv` | 2,219,201 |
| X-IIoTID (Online Resource 1, Table S7) | Al-Hawawreh et al., IEEE DataPort | `X-IIoTID_dataset.csv` | 820,834 |
| CICAPT-IIoT2024 (Online Resource 1, Table S7) | Canadian Institute for Cybersecurity, CIC APT IIoT Dataset 2024, phase-2 network data | `phase2_NetworkData.csv` | |

The paper uses a stratified 10% subsample of Edge-IIoTset (221,917 rows, `floor(10%)` of every
`Attack_type` class). Rebuild it exactly with

```bash
python src/01_make_edgeiiot_subsample.py      # writes data/raw/edgeiiot_sample2_200k.csv
```

The script selects the rows listed in `data/edgeiiot_subsample/row_indices.txt.gz` (0-based positions in the
official file, in subsample order), so no dataset content is stored in this repository.

## SHA-256 of the files used for the paper

```
f897b24578cc6fdeb3e7a0e9ff63efd5bbdc926a545abbda725e0dbb348c6bca  wustl_iiot_2021.csv
26ddc513552de36de6428b2e578efaed2b57504c716dfba847cc0109a64e1974  ton_iot_network.csv
1d3ef6c7cc22784a528a117f1158f5cad750273679437e18a39c360ee6b79fd7  DNN-EdgeIIoT-dataset.csv
ebb266ca546bcf3cc99d50cef0aa3435705fc5d717f633d58ae24b40c64056e2  edgeiiot_sample2_200k.csv (original subsample file)
```

The subsample rebuilt by `01_make_edgeiiot_subsample.py` contains the same 221,917 records. Its hash differs from
the original file, which was written by pandas: 32 `dns.qry.name` values are formatted differently
(e.g. `2.7e-05` vs `2.7000000000000002e-05`, the same float) and 5 `frame.time` strings differ only in leading
whitespace. After `02_preprocess.py` both give 221,833 records with identical values in every column (`frame.time`
equal after whitespace stripping; the timestamp parser in `09_edgeiiot_full_pipeline.py` ignores the whitespace).

## Checks after preprocessing (Table 3)

`python src/02_preprocess.py` should print:

| Dataset | Records after cleaning | Final features | Benign % | Attack % |
|---|---|---|---|---|
| WUSTL-IIoT-2021 | 1,194,464 | 41 | 92.715 | 7.285 |
| TON_IoT-Network | 190,474 | 36 | 22.071 | 77.929 |
| Edge-IIoTset (10%) | 221,833 | 49 | 72.831 | 27.169 |

## Online Resource 1, Table S7 datasets

`src/16a_prepare_xiiotid.py` and `src/16a_prepare_cicapt.py` turn the two raw files into window graphs
(`data/processed/xiiotid_windows.pkl`, `data/processed/cicapt_windows.pkl`). Expected output:

| Dataset | Windows | Attack windows | Features |
|---|---|---|---|
| X-IIoTID | 9,180 | 7,898 (86.0%) | 59 |
| CICAPT-IIoT2024 | 4,325 | 87 (2.0%) | 55 |

The X-IIoTID windows rebuilt by the script were compared with the file used for Table S7: identical hosts, masks,
labels and flow counts in every window, and identical node features. The CICAPT-IIoT2024 windows were built with
the same script on a local copy of `phase2_NetworkData.csv`; the log of that run is
`results/logs/16a_prepare_cicapt.log` (output SHA-256 `a573ef8eaf428c7267845fcba98f33cd6ff229b39454a8912895a9dc1d92ebc0`).
