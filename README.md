# PathWeaver_AE
**A High-Throughput Multi-GPU System for Graph-Based Approximate Nearest Neighbor Search**

> A Zenodo artifact for Pathweaver is available at [Zenodo](https://zenodo.org/records/15460894).

## Key Files

| File Path                               | Description                                                        |
| --------------------------------------- | ------------------------------------------------------------------ |
| `./pathweaver/csrc/pathweaver.cu`       | **CUDA kernel implementation** for PathWeaver's graph search           |
| `./pathweaver/single_pathweaver_one.py` | Python file to run PathWeaver on a **single GPU**                  |
| `./pathweaver/single_pathweaver_all.sh` | Script to run PathWeaver on a **single GPU** with parameters       |
| `./pathweaver/multi_pathweaver_one.py`  | Python file to run PathWeaver on **multiple GPUs**                 |
| `./pathweaver/multi_pathweaver_all.sh`  | Script to run PathWeaver on **multiple GPUs** with parameters      |

## Hardware Requirements
* Approximately **250 GB** of free disk space
* **4 GPUs** with at least **48 GB** of device memory
* **NVLink** bridge across GPUs
* **CUDA 12.1**

## Steps to Reproduce

### 0. Environment Setup

Install dependencies via Conda:

```bash
conda env create -f environment.yaml
conda activate pathweaver
```

### 1. Build PathWeaver Binary

```bash
cd pathweaver
bash build.sh
```

### 2. Run Single-GPU Artifact Evaluation

```bash
bash single_pathweaver_all.sh
```

### 3. Run Multi-GPU Artifact Evaluation

```bash
bash multi_pathweaver_all.sh
```

## Acknowledgment

This project is built upon and extends the **CAGRA method** from the [RAPIDS cuVS](https://github.com/rapidsai/cuvs) library, an efficient GPU-accelerated framework for vector search.
