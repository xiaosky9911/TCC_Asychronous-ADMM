# Asynchronous Edge-Consensus ADMM for Distributed Cloud Computing Resource Scheduling

This repository contains the source code and experimental data for the paper:

> **"[Paper Title]"**  
> Submitted to *IEEE Transactions on Cloud Computing*

---

## Overview

This project implements a distributed computing resource scheduling framework based on **edge-consensus ADMM** (Alternating Direction Method of Multipliers), supporting both **synchronous** and **asynchronous** execution modes. The system is designed for multi-region cloud computing environments where each region contains multiple heterogeneous computing nodes. Communication is implemented via **MPI** (Message Passing Interface), and local subproblems are solved using **Gurobi**.

The primary scenario studied is an **8-region system with 24 computing nodes** (3 nodes per region), with scalability experiments covering 4-, 8-, 16-, and 32-node configurations.

---

## Requirements

| Dependency   | Version (tested) | Notes                                                   |
|--------------|------------------|---------------------------------------------------------|
| Python       | ≥ 3.8            |                                                         |
| mpi4py       | ≥ 3.1            | Requires an MPI runtime (e.g., Microsoft MPI on Windows, OpenMPI on Linux) |
| gurobipy     | ≥ 10.0           | A valid Gurobi license is required                      |
| pandas       | ≥ 1.3            |                                                         |
| numpy        | ≥ 1.21           |                                                         |
| matplotlib   | ≥ 3.4            |                                                         |
| networkx     | ≥ 2.6            | Only for topology visualization                         |
| openpyxl     | ≥ 3.0            | For reading `.xlsx` data files                          |

Install Python dependencies:
```bash
pip install mpi4py gurobipy pandas numpy matplotlib networkx openpyxl
```

---

## Repository Structure

```
.
├── README.md
├── run_custom.py                   # Main launcher: configure parameters and run MPI tasks
├── RE_Asynchronous.py              # Asynchronous edge-consensus ADMM scheduler
├── RE_Synchronous.py               # Synchronous edge-consensus ADMM scheduler
├── Centralized_cost.py             # Centralized Gurobi baseline (optimal solution)
├── n16_topology.py                 # Network topology visualization for 16-node system
├── plot_convergence_compare.py     # Convergence comparison plots (obj. value vs. time/iterations)
├── Scaliability.py                 # Scalability bar chart (wall time / iterations / messages)
│
├── need_data_N4.xlsx               # Dataset: 4 nodes (2 regions × 2–3 nodes)
├── need_data_N8.xlsx               # Dataset: 8 nodes (8 regions × 3 nodes)  ← main experiment
├── need_data_N16.xlsx              # Dataset: 16 nodes (16 regions × 3 nodes)
├── need_data_N32.xlsx              # Dataset: 32 nodes (32 regions × 3 nodes)
│
├── CASE1_FIX/                      # Experiment: fixed-straggler delay scenario
│   ├── admm_trace_sync.csv
│   ├── admm_trace_async.csv
│   ├── admm_trace_async_undelay.csv
│   ├── admm_trace_async_delay.csv
│   ├── plot_convergence_compare.py
│   └── plot_convergence_vertical.py
│
└── CASE1_STO/                      # Experiment: stochastic-delay scenario
    ├── admm_trace_sync.csv
    ├── admm_trace_async.csv
    ├── admm_trace_async_undelay.csv
    ├── admm_trace_async_delay.csv
    └── plot_convergence_vertical.py
```

---

## Dataset Files (`.xlsx`)

Each `need_data_N{K}.xlsx` file describes a K-node computing system. It contains three sheets:

| Sheet Name   | Description                                                                 |
|--------------|-----------------------------------------------------------------------------|
| `城市参数`   | Per-node city parameters (e.g., computing capacity, task demand, cost coefficients). Each row corresponds to one computing node; columns 2–9 are numerical parameters. |
| `带宽上限`   | K×K bandwidth upper-limit matrix (Gbps). Entry (i,j) is the maximum transmission bandwidth between node i and node j. |
| `单价系数`   | K×K unit cost coefficient matrix. Entry (i,j) is the per-unit transmission cost between node i and node j. |

The number of computing nodes per region for each dataset:

| Dataset         | Regions (I) | Nodes per Region (J)    | Total Nodes |
|-----------------|-------------|-------------------------|-------------|
| need_data_N4    | 4           | [3, 3, 2, 2]            | 10          |
| need_data_N8    | 8           | [3, 3, 3, 3, 3, 3, 3, 3]| 24          |
| need_data_N16   | 16          | [3] × 16                | 48          |
| need_data_N32   | 32          | [3] × 32                | 96          |

> **Note:** The MPI process count (`-np`) must equal the number of *regions* I, not the total number of computing nodes.

---

## Program Files

### `run_custom.py` — Main Configuration and Launcher

The recommended entry point. Edit the parameters at the top of this file, then run:
```bash
python run_custom.py
```
It automatically injects environment variables and invokes `mpiexec`.

### `RE_Asynchronous.py` — Asynchronous ADMM

Implements the **asynchronous edge-consensus ADMM** algorithm with straggler tolerance. Each MPI process corresponds to one region. Processes exchange auxiliary variables (u) with neighbors asynchronously, subject to a minimum dependency set constraint (S) and a maximum lag bound (τ). Local subproblems are solved with Gurobi.

### `RE_Synchronous.py` — Synchronous ADMM

Implements the **synchronous edge-consensus ADMM** baseline. All processes must complete each iteration before the next round begins (BSP barrier). Used for convergence and scalability comparison.

### `Centralized_cost.py` — Centralized Baseline

Solves the global optimization problem in a centralized manner using Gurobi. The resulting optimal objective value is used as the reference (ground truth) for measuring convergence error.

```bash
python Centralized_cost.py need_data_N8
```

### `n16_topology.py` — Topology Visualization

Reads `need_data_N16.xlsx` and draws a weighted undirected graph of the 16-node network topology, split into two clusters. Outputs `n16_topology.png`.

```bash
python n16_topology.py
```

### `plot_convergence_compare.py` — Convergence Plots

Reads CSV trace files and plots objective-value convergence curves for sync-ADMM and three async-ADMM variants vs. wall-clock time or iteration count.

```bash
# Use default CSV files in the current directory
python plot_convergence_compare.py

# Specify custom CSV paths
python plot_convergence_compare.py sync.csv async.csv async_undelay.csv async_delay.csv
```

### `Scaliability.py` — Scalability Bar Chart

Plots the percentage reduction of the async algorithm over sync for wall-clock time, iteration count, and message count across 8-, 16-, and 32-node systems.

```bash
python Scaliability.py
```

---

## Parameters

All algorithmic parameters are configured in `run_custom.py` and passed to the solver scripts as **environment variables**.

### Basic Configuration

| Parameter     | Variable in `run_custom.py` | Description                                                    |
|---------------|-----------------------------|----------------------------------------------------------------|
| Node count    | `NUM_NODES`                 | Number of MPI processes (= number of regions I). Must match dataset: 4, 8, 16, or 32. |
| Run mode      | `MODE`                      | `"sync"`: synchronous ADMM; `"async"`: asynchronous ADMM.     |
| Dataset       | `DATASET`                   | Auto-set from `NUM_NODES`. E.g., `"need_data_N8"`.            |

### Common Parameters (Sync & Async)

| Environment Variable   | Default  | Description                                                                  |
|------------------------|----------|------------------------------------------------------------------------------|
| `SCENARIO`             | `"2"`    | Delay scenario: `1` = fixed two slow nodes; `2` = Gaussian jitter; `3` = lognormal jitter. |
| `MAX_ITER`             | `3500`   | Maximum number of ADMM iterations.                                           |
| `RHO_STRATEGY`         | `"fixed"`| Penalty parameter strategy: `fixed` / `adaptive` / `hybrid` / `staged_fixed`. |
| `RHO_FIXED`            | `160`    | Value of ρ when `RHO_STRATEGY = "fixed"`.                                    |
| `PROX_RATIO`           | `1.5`    | Proximal term scaling factor in the local subproblem.                        |
| `SLOW_RATIO`           | `0.25`   | Fraction of straggler nodes (used in `SCENARIO = 1`).                       |
| `SLOW_SLEEP`           | `0.04`   | Artificial sleep duration (s) for straggler nodes.                           |
| `NORMAL_SLEEP`         | `0.005`  | Artificial sleep duration (s) for normal nodes.                              |
| `CONV_EPS_ABS`         | `0.01`   | Absolute convergence threshold ε_abs.                                        |
| `CONV_EPS_REL`         | `0.01`   | Relative convergence threshold ε_rel.                                        |
| `CONV_STABLE_NEED`     | `6`      | Number of consecutive rounds that must satisfy the convergence condition.    |
| `CENTRALIZED_BASE`     | (auto)   | Centralized optimal objective value, used to compute relative error.         |

### Delay Scenario Parameters

| Environment Variable    | Default | Applicable Scenario | Description                                              |
|-------------------------|---------|---------------------|----------------------------------------------------------|
| `CONGESTION_JITTER_STD` | `0.003` | SCENARIO = 2        | Standard deviation (s) of Gaussian communication jitter. |
| `LOGNORMAL_MEAN`        | `-4.8`  | SCENARIO = 3        | Log-normal distribution μ parameter for delay jitter.   |
| `LOGNORMAL_SIGMA`       | `1.0`   | SCENARIO = 3        | Log-normal distribution σ parameter for delay jitter.   |

### Asynchronous-Only Parameters

| Environment Variable | Default                         | Description                                                                    |
|----------------------|---------------------------------|--------------------------------------------------------------------------------|
| `MIN_DEP_SET` (S)    | `⌈(1−0.25)×\|neighbors\|⌉`    | Minimum number of neighbors whose updated messages must be received before performing a local update. Tolerates up to `|neighbors| − S` stragglers. |
| `MAX_LAG` (τ)        | `⌈T_slow / T_fast⌉`            | Maximum staleness bound. An update from neighbor j is discarded if its iteration lag exceeds τ. |

---

## Usage

### Step 1: Verify the centralized optimal value

```bash
python Centralized_cost.py need_data_N8
```

Note the printed objective value (e.g., `1182823039.614968`) and set `centralized_base` in `run_custom.py`.

### Step 2: Configure `run_custom.py`

Edit the top section of `run_custom.py`:

```python
NUM_NODES = 8        # 8 regions, matching need_data_N8
MODE = "async"       # "sync" or "async"

ENV_VARS = {
    "SCENARIO":      "2",     # Gaussian-jitter scenario
    "MAX_ITER":      "3500",
    "RHO_FIXED":     "160",
    ...
}
```

### Step 3: Run

```bash
python run_custom.py
```

This executes:
```bash
mpiexec -np 8 python RE_Asynchronous.py need_data_N8
```

### Step 4: Reproduce paper figures

**Convergence comparison (Section V-B):**
```bash
cd CASE1_FIX
python plot_convergence_compare.py
```

**Scalability experiment (Section V-C):**
```bash
python Scaliability.py
```

---

## Output Files

| File                                | Description                                             |
|-------------------------------------|---------------------------------------------------------|
| `admm_trace_async.csv`              | Per-iteration log: objective value, primal/dual residuals, wall-clock time. |
| `admm_async_auxu_metrics.csv`       | Convergence metrics summary.                            |
| `admm_async_auxu_obj_trace.csv`     | Objective value trace.                                  |
| `admm_async_auxu_rank_summary.csv`  | Per-rank iteration and message statistics.              |
| `n16_topology.png`                  | 16-node network topology diagram.                       |

---

## Centralized Baseline Values

Pre-computed optimal objective values for each dataset (used as `CENTRALIZED_BASE`):

| Dataset       | Optimal Objective Value    |
|---------------|----------------------------|
| need_data_N4  | 481,369,566.306250         |
| need_data_N8  | 1,182,823,039.614968       |
| need_data_N16 | 2,459,936,083.674638       |
| need_data_N32 | 4,910,228,836.029585       |

---

## License

This code is released for academic use. If you use this code or data in your research, please cite:

```bibtex
@misc{async_admm_cloud,
  author       = {[Author Names]},
  title        = {Asynchronous Edge-Consensus ADMM for Distributed Cloud Computing Resource Scheduling},
  year         = {2025},
  howpublished = {\url{https://github.com/[username]/[repo-name]}},
  note         = {Source code for the paper submitted to IEEE Transactions on Cloud Computing}
}
```
