# Asynchronous Edge-Consensus ADMM for Geo-Distributed Cloud Scheduling

This repository contains the source code, datasets, and experiment wrappers for:

> **Geo-Distributed Workload Scheduling of Cloud Data Centers through Consensus-Based Asynchronous Decentralized Algorithm**

The implementation compares synchronous and asynchronous edge-consensus ADMM for multi-region cloud resource scheduling. Each MPI process represents one region. Local mixed-integer subproblems are solved with Gurobi, and neighboring regions exchange edge-consensus auxiliary variables through MPI.

---

## What Is Included

- Synchronous and asynchronous edge-consensus ADMM solvers.
- Centralized Gurobi baseline for objective-value comparison.
- Excel datasets for 4-, 8-, 16-, and 32-region systems.
- Configurable delay models, topology options, dependency-set rules, lag bounds, and convergence tolerances.
- Reproduction scripts for convergence comparison, scalability analysis, delay sensitivity, dependency-set sensitivity, and initialization-transient analysis.

---

## Requirements

| Dependency | Tested Version | Notes |
|---|---:|---|
| Python | >= 3.8 | |
| mpi4py | >= 3.1 | Requires an MPI runtime such as MPICH, OpenMPI, or Microsoft MPI |
| gurobipy | >= 10.0 | A valid Gurobi license is required |
| pandas | >= 1.3 | |
| numpy | >= 1.21 | |
| matplotlib | >= 3.4 | Required for figures |
| networkx | >= 2.6 | Only for topology visualization |
| openpyxl | >= 3.0 | Required for reading `.xlsx` datasets |

Install Python packages:

```bash
pip install mpi4py gurobipy pandas numpy matplotlib networkx openpyxl
```

Make sure `mpiexec` is available. If it is not on `PATH`, the experiment wrappers try:

```text
~/Library/Python/3.9/bin/mpiexec
```

You can also override paths:

```bash
MPIEXEC=/path/to/mpiexec PYTHON=/path/to/python python run_custom.py
```

---

## Repository Layout

```text
.
├── RE_Asynchronous.py                    # Asynchronous edge-consensus ADMM
├── RE_Synchronous.py                     # Synchronous edge-consensus ADMM
├── Centralized_cost.py                   # Centralized Gurobi baseline
├── run_custom.py                         # General launcher for sync/async runs
├── plot_convergence_compare.py           # Objective convergence plotter
├── Scaliability.py                       # Scalability comparison figure
├── n16_topology.py                       # 16-region topology visualization
├── run_irregular16_s_sensitivity.py      # Minimum dependency set sensitivity
├── run_irregular16_tau_sensitivity.py    # Maximum lag bound sensitivity
├── run_init_convergence_experiment.py    # Zero-u initialization transient experiment
├── need_data_N4.xlsx
├── need_data_N8.xlsx
├── need_data_N16.xlsx
└── need_data_N32.xlsx
```

---

## Datasets

Each `need_data_N*.xlsx` file contains:

| Sheet | Meaning |
|---|---|
| `城市参数` | Per-city demand, capacity, and cost parameters |
| `带宽上限` | Region-to-region bandwidth upper bounds |
| `单价系数` | Region-to-region transmission price coefficients |

Dataset sizes:

| Dataset | MPI Regions | Cities per Region | Total Cities |
|---|---:|---|---:|
| `need_data_N4` | 4 | `[3, 3, 2, 2]` | 10 |
| `need_data_N8` | 8 | `[3] x 8` | 24 |
| `need_data_N16` | 16 | `[3] x 16` | 48 |
| `need_data_N32` | 32 | `[3] x 32` | 96 |

Important: the MPI process count must equal the number of regions, not the number of cities. For example, run `need_data_N8` with `-np 8`.

---

## Quick Start

### 1. Centralized Baseline

```bash
python Centralized_cost.py need_data_N8
```

This prints the centralized objective value used as the reference for relative gaps.

### 2. General Sync/Async Run

Edit the top of `run_custom.py`, especially:

```python
NUM_NODES = 8
MODE = "async"   # or "sync"
```

Then run:

```bash
python run_custom.py
```

This launches either:

```bash
mpiexec -np 8 python RE_Asynchronous.py need_data_N8
```

or:

```bash
mpiexec -np 8 python RE_Synchronous.py need_data_N8
```

---

## Core Parameters

Parameters are passed through environment variables.

| Variable | Typical Value | Description |
|---|---:|---|
| `MAX_ITER` | `3500` | Maximum local ADMM iterations |
| `RHO_STRATEGY` | `fixed` | `fixed`, `adaptive`, `hybrid`, or `staged_fixed` |
| `RHO_FIXED` | `160` | Penalty parameter when using fixed rho |
| `PROX_RATIO` | `1.5` | Proximal term coefficient relative to rho |
| `CONV_EPS_ABS` | `0.01` | Absolute convergence tolerance |
| `CONV_EPS_REL` | `0.01` | Relative convergence tolerance |
| `CONV_STABLE_NEED` | `6` | Required consecutive converged rounds before local stop |
| `CENTRALIZED_BASE` | dataset-specific | Centralized objective used for gap computation |
| `BASE_DIR` | repo root | Directory for datasets and generated outputs |

### Asynchronous Parameters

| Variable | Typical Value | Description |
|---|---:|---|
| `S_MODE` | `fixed` or `formula_alpha` | How to choose the minimum dependency set |
| `MIN_DEP_SET` | `6` | Fixed minimum number of fresh neighbor messages |
| `ALPHA_DEP` | `0.75` | Used by `S_MODE=formula_alpha`, with `s_i=ceil(alpha \|N_i\|)` |
| `MAX_LAG` | `16` | Maximum accepted neighbor staleness |
| `TOPOLOGY` | `full` or JSON | Region graph. JSON maps each region to neighbor indices |

### Delay Scenarios

| `SCENARIO` | Meaning |
|---:|---|
| `0` | Normal-speed updates with Gaussian jitter |
| `1` | Fixed two slow ranks, used by legacy case studies |
| `2` | Heterogeneous base delay plus lognormal jitter |
| `3` | Normal update delay with random non-stop message loss during communication |
| `4` | Fixed-straggler setting |
| `5` | Fixed straggler identity plus relative Gaussian perturbation |
| `6` | Fixed straggler identity plus uniform random update time |

### Delay Scenario Parameters

| Variable | Default | Used By | Description |
|---|---:|---|---|
| `NORMAL_SLEEP` | `0.005` | `0`, `1`, `4`, `5`, fallback | Base update delay for normal regions, in seconds |
| `SLOW_SLEEP` | `0.04` | `1`, `4`, `5` | Base update delay for slow regions, in seconds |
| `SLOW_RATIO` | `0.25` | `4`, `5`, `6` | Fraction of regions treated as fixed stragglers. The current implementation selects the last `ceil(SLOW_RATIO * I)` ranks |
| `CONGESTION_JITTER_STD` | `0.003` | `0` | Standard deviation of Gaussian jitter added to `NORMAL_SLEEP` |
| `LOGNORMAL_MEAN` | `-4.8` | `2` | Mean parameter of the lognormal jitter distribution |
| `LOGNORMAL_SIGMA` | `0.9` | `2` | Sigma parameter of the lognormal jitter distribution |
| `RANDOM_DELAY_CV` | `0.3` | `5` | Relative Gaussian perturbation strength around each rank's base delay |
| `NORMAL_SLEEP_LOW` | `0.004` | `6` | Lower bound of uniform update delay for normal regions |
| `NORMAL_SLEEP_HIGH` | `0.006` | `6` | Upper bound of uniform update delay for normal regions |
| `SLOW_SLEEP_LOW` | `0.032` | `6` | Lower bound of uniform update delay for fixed stragglers |
| `SLOW_SLEEP_HIGH` | `0.048` | `6` | Upper bound of uniform update delay for fixed stragglers |

Scenario behavior:

| `SCENARIO` | Effective Delay / Communication Model |
|---:|---|
| `0` | `sleep = NORMAL_SLEEP + abs(N(0, CONGESTION_JITTER_STD))` for every rank |
| `1` | Rank `2` sleeps `SLOW_SLEEP`; rank `5` sleeps `2 * SLOW_SLEEP`; all other ranks sleep `NORMAL_SLEEP` |
| `2` | Each rank uses a fixed base delay and multiplier, plus lognormal jitter: `sleep = factor_rank * (base_rank + LogNormal(LOGNORMAL_MEAN, LOGNORMAL_SIGMA))` |
| `3` | Sleep falls back to `NORMAL_SLEEP`; during message broadcast, each non-stop message has 5% probability of being skipped before the last iteration |
| `4` | Fixed stragglers sleep `SLOW_SLEEP`; normal ranks sleep `NORMAL_SLEEP` |
| `5` | Same fixed straggler identities as scenario 4, but with relative jitter: `sleep = base_delay + base_delay * abs(N(0, RANDOM_DELAY_CV))` |
| `6` | Fixed stragglers sample `U(SLOW_SLEEP_LOW, SLOW_SLEEP_HIGH)`; normal ranks sample `U(NORMAL_SLEEP_LOW, NORMAL_SLEEP_HIGH)` |

The fixed-straggler example configuration is:

```text
SCENARIO=4
NORMAL_SLEEP=0.005
SLOW_SLEEP=0.04
SLOW_RATIO=0.25
```

---

## Reproducing Experiments

The scripts below provide ready-to-run experiment configurations. They can be used as-is to reproduce the included tables and figures, or edited to explore other datasets, topologies, delay models, and algorithm parameters.

### 1. Minimum Dependency Set Sensitivity

```bash
python run_irregular16_s_sensitivity.py
```

Setting:

- Dataset: `need_data_N16`
- Topology: irregular 16-region sparse graph embedded in the script
- Delay model: fixed stragglers, `SCENARIO=4`
- Fixed lag: `MAX_LAG=16`
- Tested dependency settings: `s=2`, `alpha=0.25`, `alpha=0.50`, `alpha=0.75`, `alpha=1.00`
- Repetitions: `N_TRIALS=5` by default

The wrapper prints the averaged table directly and removes temporary solver files after each trial. It does not save intermediate CSV tables.

Optional:

```bash
N_TRIALS=3 FIXED_TAU=8 python run_irregular16_s_sensitivity.py
```

### 2. Maximum Lag Bound Sensitivity

```bash
python run_irregular16_tau_sensitivity.py
```

Setting:

- Dataset: `need_data_N16`
- Topology: irregular 16-region sparse graph
- Delay model: fixed stragglers, `SCENARIO=4`
- Dependency rule: `s_i=ceil(0.75 |N_i|)`
- Tested `tau`: `0, 4, 8, 16, 32, 64`
- Repetitions: `N_TRIALS=5` by default

The wrapper prints the averaged table directly and removes temporary solver files after each trial.

Optional:

```bash
N_TRIALS=3 ALPHA_DEP=0.5 python run_irregular16_tau_sensitivity.py
```

### 3. Zero-u Initialization Transient Experiment

```bash
python run_init_convergence_experiment.py
```

This experiment addresses the transient behavior under zero initialization of the auxiliary variable `u` when the initial workload distribution is imbalanced.

Setting:

- Dataset: `need_data_N8`
- Topology: full graph
- Delay model: fixed stragglers, `SCENARIO=4`
- `U_INIT_MODE=zero`
- `MAX_ITER=500`
- `MAX_LAG=16`
- `ALPHA_DEP=0.75`
- Demand is rescaled while total workload is preserved

Workload cases:

| Case | Target Workload Distribution |
|---|---|
| `uniform` | Equal share across all 8 regions |
| `half_spread` | 80% demand in the first half of regions, 20% in the rest |
| `concentrated` | 80% demand in one region, 20% shared by the others |

Output:

```text
init_convergence.png
```

The figure contains:

- normalized primal fixed-point residual, `norm_dx_global`;
- normalized dual fixed-point residual, `norm_du_global`;
- a horizontal convergence threshold at `y=1`.

The wrapper reads solver trace files in memory, deletes temporary CSV/PKL outputs, and saves only the final figure.

Optional:

```bash
INIT_CASES=uniform,concentrated INIT_MAX_ITER=800 python run_init_convergence_experiment.py
```

---

## Solver Output Files

When running `RE_Asynchronous.py` or `RE_Synchronous.py` directly, the solver writes diagnostic files in the repository root.

Common asynchronous outputs:

| File | Description |
|---|---|
| `admm_trace_async.csv` | Global per-iteration trace |
| `admm_async_auxu_metrics.csv` | One-row summary of objective, gap, runtime, messages, and consensus residual |
| `admm_async_auxu_rank_summary.csv` | Per-rank runtime, iteration, and message statistics |
| `admm_async_auxu_obj_trace.csv` | Per-rank local objective trace |
| `admm_async_auxu_final_x.pkl` | Final local edge variables for offline diagnostics |

Important trace columns:

| Column | Meaning |
|---|---|
| `dx_global` | Raw fixed-point change in local edge variables |
| `du_global` | Raw fixed-point change in auxiliary variables |
| `eps_x_global` | Dynamic primal threshold |
| `eps_u_global` | Dynamic dual threshold |
| `norm_dx_global` | `dx_global / eps_x_global` |
| `norm_du_global` | `du_global / eps_u_global` |
| `obj_global` | Sum of local physical objectives |

Note: in the auxiliary-variable implementation, `dx_global` and `du_global` are fixed-point residuals. They are useful convergence diagnostics but are not exactly the standard ADMM primal and dual residuals.

---

## Plotting Existing Traces

Objective convergence comparison:

```bash
python plot_convergence_compare.py
```

or with explicit files:

```bash
python plot_convergence_compare.py sync.csv async.csv async_undelay.csv async_delay.csv
```

Scalability figure:

```bash
python Scaliability.py
```

16-region topology figure:

```bash
python n16_topology.py
```

---

## Centralized Baseline Values

The default values used by `run_custom.py` are:

| Dataset | Objective |
|---|---:|
| `need_data_N4` | 481,369,566.306250 |
| `need_data_N8` | 1,182,823,039.614968 |
| `need_data_N16` | 2,459,936,083.674638 |
| `need_data_N32` | 4,910,228,836.029585 |

When demand scaling is enabled, for example in `run_init_convergence_experiment.py`, the centralized baseline is recomputed for each scaled demand profile.

---

## Troubleshooting

### `mpiexec` cannot bind or create a PMI port

Some sandboxed environments block local port binding. Run the experiment in a normal terminal or allow `mpiexec` to run outside the sandbox.

### Gurobi returns status 4 after demand scaling

Scaled demand can become fractional while local decision variables are integer. The initialization experiment uses:

```text
ROUND_SCALED_DEMAND=1
```

so centralized and distributed solvers both use rounded scaled demand.

### Raw residuals look very large

The raw auxiliary-variable residual `du_global` can be large because `u` is scaled by `rho`. For convergence plots, prefer:

```text
norm_dx_global
norm_du_global
```

Values below `1` are below the dynamic convergence threshold.

---

## Citation

If you use this code or data in your research, please cite:

```bibtex
@misc{async_admm_cloud,
  author       = {Chen, Mengxiao and others},
  title        = {Geo-Distributed Workload Scheduling of Cloud Data Centers through Consensus-Based Asynchronous Decentralized Algorithm},
  year         = {2025},
  howpublished = {\url{https://github.com/xiaosky9911/TCC_Asychronous-ADMM}},
  note         = {Source code and experiment data}
}
```

---

## License

This repository is released for academic research use.
