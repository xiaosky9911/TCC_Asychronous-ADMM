# -*- coding: utf-8 -*-
"""Sensitivity analysis for the maximum delay bound tau.

The experiment uses the irregular 16-region topology and a degree-proportional
minimum dependency set s_i = ceil(alpha |N_i|).  The delay model is the fixed
straggler scenario: normal schedulers use a 0.005 s update delay, while the
fixed straggler schedulers use a 0.04 s update delay.
"""

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
PYTHON = os.environ.get("PYTHON", "/usr/bin/python3")
MPIEXEC = (
    os.environ.get("MPIEXEC")
    or shutil.which("mpiexec")
    or str(Path.home() / "Library" / "Python" / "3.9" / "bin" / "mpiexec")
)
ASYNC_SCRIPT = BASE_DIR / "RE_Asynchronous.py"
CENTRALIZED_SCRIPT = BASE_DIR / "Centralized_cost.py"

DATASET = "need_data_N16"
NUM_NODES = 16
N_TRIALS = int(os.environ.get("N_TRIALS", "5"))
ALPHA_DEP = float(os.environ.get("ALPHA_DEP", "0.75"))
TAUS = [0, 4, 8, 16, 32, 64]

IRREGULAR16_TOPOLOGY = {
    "0": [1, 2, 3, 5, 8, 12],
    "1": [0, 2],
    "2": [0, 1, 4, 6],
    "3": [0, 4, 7, 10],
    "4": [2, 3, 5],
    "5": [0, 4, 6, 11],
    "6": [2, 5, 7],
    "7": [3, 6, 15],
    "8": [0, 9, 10, 12, 13],
    "9": [8, 10],
    "10": [3, 8, 9, 11, 14],
    "11": [5, 10, 12],
    "12": [0, 8, 11, 13, 15],
    "13": [8, 12, 14],
    "14": [10, 13, 15],
    "15": [7, 12, 14],
}

COMMON_ENV = {
    "BASE_DIR": str(BASE_DIR),
    "TOPOLOGY": json.dumps(IRREGULAR16_TOPOLOGY),
    "SCENARIO": "4",
    "MAX_ITER": "3500",
    "S_MODE": "formula_alpha",
    "ALPHA_DEP": str(ALPHA_DEP),
    "RHO_STRATEGY": "fixed",
    "RHO_FIXED": "160",
    "PROX_RATIO": "1.5",
    "NORMAL_SLEEP": "0.005",
    "SLOW_SLEEP": "0.04",
    "SLOW_RATIO": "0.25",
    "CONGESTION_JITTER_STD": "0.003",
    "CONV_EPS_ABS": "0.01",
    "CONV_EPS_REL": "0.01",
    "CONV_STABLE_NEED": "6",
    "GRB_THREADS": "1",
}

TEMP_OUTPUTS = [
    "admm_async_auxu_metrics.csv",
    "admm_async_auxu_rank_summary.csv",
    "admm_async_auxu_obj_trace.csv",
    "admm_trace_async.csv",
    "admm_trace_async_delay.csv",
    "admm_trace_async_undelay.csv",
    "admm_async_auxu_final_x.pkl",
]


def parse_centralized(text):
    for line in text.splitlines():
        if "总目标函数值" in line or "模型 objective" in line:
            return float(line.split(":")[-1].strip())
    raise RuntimeError("Failed to parse centralized objective:\n" + text[-2000:])


def cleanup_temp_outputs():
    for name in TEMP_OUTPUTS:
        path = BASE_DIR / name
        if path.exists():
            path.unlink()


def compute_centralized():
    env = os.environ.copy()
    env.update({"BASE_DIR": str(BASE_DIR), "TOPOLOGY": json.dumps(IRREGULAR16_TOPOLOGY)})
    result = subprocess.run(
        [PYTHON, str(CENTRALIZED_SCRIPT), DATASET],
        env=env,
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )
    return parse_centralized(result.stdout + result.stderr)


def run_one(tau, trial, centralized_base):
    if MPIEXEC is None:
        raise RuntimeError("mpiexec was not found. Set MPIEXEC=/path/to/mpiexec.")

    env = os.environ.copy()
    env.update(COMMON_ENV)
    env["MAX_LAG"] = str(tau)
    env["CENTRALIZED_BASE"] = str(centralized_base)

    started = time.time()
    cmd = [MPIEXEC, "-np", str(NUM_NODES), PYTHON, str(ASYNC_SCRIPT), DATASET]
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=1200)

    metrics_path = BASE_DIR / "admm_async_auxu_metrics.csv"
    if not metrics_path.exists() or metrics_path.stat().st_mtime < started:
        raise RuntimeError(
            f"Run failed for tau={tau}, trial {trial}:\n"
            + (result.stdout + result.stderr)[-3000:]
        )

    metrics = pd.read_csv(metrics_path, encoding="utf-8-sig").iloc[0].to_dict()
    row = {
        "tau": tau,
        "trial": trial,
        "alpha": ALPHA_DEP,
        "obj_total": metrics["obj_total"],
        "rel_gap_pct": metrics["rel_diff_to_centralized"] * 100.0,
        "wall_clock_total_s": metrics["wall_clock_total"],
        "avg_iter": metrics["avg_iter"],
        "avg_msg_send_cnt": metrics["avg_msg_send_cnt"],
        "iter_balance_pct": metrics["iter_balance_pct"],
        "offline_consensus_residual": metrics["offline_primal_consensus_residual"],
    }
    print(
        f"tau={tau:>2d} trial {trial}: gap={row['rel_gap_pct']:.4f}%, "
        f"time={row['wall_clock_total_s']:.3f}s, iter={row['avg_iter']:.1f}"
    )
    cleanup_temp_outputs()
    return row


def main():
    centralized_base = compute_centralized()
    rows = []
    for tau in TAUS:
        for trial in range(1, N_TRIALS + 1):
            rows.append(run_one(tau, trial, centralized_base))

    raw = pd.DataFrame(rows)

    summary = (
        raw.groupby(["tau", "alpha"], as_index=False)
        .agg(
            rel_gap_pct=("rel_gap_pct", "mean"),
            rel_gap_pct_std=("rel_gap_pct", "std"),
            wall_clock_total_s=("wall_clock_total_s", "mean"),
            wall_clock_total_s_std=("wall_clock_total_s", "std"),
            avg_iter=("avg_iter", "mean"),
            avg_msg_send_cnt=("avg_msg_send_cnt", "mean"),
            iter_balance_pct=("iter_balance_pct", "mean"),
            offline_consensus_residual=("offline_consensus_residual", "mean"),
            obj_total=("obj_total", "mean"),
        )
        .sort_values("tau")
    )

    display = summary.rename(
        columns={
            "tau": r"$\tau$",
            "rel_gap_pct": "Gap (%)",
            "rel_gap_pct_std": "Gap std. (%)",
            "wall_clock_total_s": "Wall-clock (s)",
            "wall_clock_total_s_std": "Time std. (s)",
            "avg_iter": "Avg. iter.",
            "avg_msg_send_cnt": "Avg. messages",
            "iter_balance_pct": "Iter. imbalance (%)",
            "offline_consensus_residual": "Consensus residual",
        }
    )
    display_cols = [
        r"$\tau$",
        "Gap (%)",
        "Wall-clock (s)",
        "Avg. iter.",
        "Avg. messages",
        "Iter. imbalance (%)",
        "Consensus residual",
    ]

    print("\nFixed-straggler tau sensitivity, averaged over "
          f"{N_TRIALS} trials (alpha={ALPHA_DEP:.2f})")
    print(display[display_cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
