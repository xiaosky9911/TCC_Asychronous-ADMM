# -*- coding: utf-8 -*-
"""Initial workload-distribution convergence experiment.

This script examines whether zero initialization of the auxiliary variable u
causes problematic transient behavior when the initial workload distribution is
imbalanced.  It saves only the final convergence plot; intermediate solver CSV
files are read and then removed.
"""

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
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

DATASET = os.environ.get("INIT_DATASET", "need_data_N8")
NUM_NODES = int(os.environ.get("INIT_NUM_NODES", "8"))
INIT_MAX_ITER = os.environ.get("INIT_MAX_ITER", "500")
RUN_TIMEOUT = int(os.environ.get("RUN_TIMEOUT", "300"))
PLOT_PATH = BASE_DIR / os.environ.get("INIT_PLOT_PATH", "init_convergence.png")
PLOT_Y_MIN = float(os.environ.get("INIT_PLOT_Y_MIN", "1e-1"))
PLOT_Y_MAX = float(os.environ.get("INIT_PLOT_Y_MAX", "1e2"))

COMMON_ENV = {
    "BASE_DIR": str(BASE_DIR),
    "TOPOLOGY": "full",
    "SCENARIO": "4",
    "MAX_ITER": INIT_MAX_ITER,
    "S_MODE": "formula_alpha",
    "ALPHA_DEP": "0.75",
    "MAX_LAG": "16",
    "RHO_STRATEGY": "fixed",
    "RHO_FIXED": "160",
    "PROX_RATIO": "1.5",
    "NORMAL_SLEEP": "0.005",
    "SLOW_SLEEP": "0.04",
    "SLOW_RATIO": "0.25",
    "CONV_EPS_ABS": "0.01",
    "CONV_EPS_REL": "0.01",
    "CONV_STABLE_NEED": "6",
    "U_INIT_MODE": "zero",
    "ROUND_SCALED_DEMAND": "1",
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
    "admm_metrics_init_conv_uniform.csv",
    "admm_metrics_init_conv_half_spread.csv",
    "admm_metrics_init_conv_concentrated.csv",
    "admm_trace_init_conv_uniform.csv",
    "admm_trace_init_conv_half_spread.csv",
    "admm_trace_init_conv_concentrated.csv",
    "init_convergence_summary.csv",
]


def dataset_region_sizes():
    if DATASET == "need_data_N4":
        return [3, 3, 2, 2]
    if DATASET in {"need_data_N8", "need_data_N16", "need_data_N32"}:
        return [3] * NUM_NODES
    raise ValueError(f"Unsupported INIT_DATASET={DATASET}")


def load_base_region_workloads():
    """Return original per-region workload, using the same demand columns as solvers."""
    data_path = BASE_DIR / f"{DATASET}.xlsx"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing dataset file: {data_path}")

    region_sizes = dataset_region_sizes()
    fz = pd.read_excel(data_path, sheet_name="城市参数")
    qy = fz.values[0:sum(region_sizes), 1:9]

    workloads = []
    start = 0
    for size in region_sizes:
        rows = qy[start:start + size]
        workloads.append(float(rows[:, 0:3].sum()))
        start += size
    return workloads


def scales_for_target_shares(target_shares):
    base_workloads = load_base_region_workloads()
    total_workload = sum(base_workloads)
    if len(target_shares) != NUM_NODES:
        raise ValueError("target_shares length must equal INIT_NUM_NODES")

    scale = []
    for base, share in zip(base_workloads, target_shares):
        target = total_workload * share
        scale.append(target / base if base > 0 else 1.0)
    return scale


def build_workload_cases():
    """Build scale factors so total workload is unchanged across all scenarios."""
    n = NUM_NODES
    half = n // 2
    if n != 8:
        raise ValueError("This initialization experiment is defined for INIT_NUM_NODES=8.")

    uniform = [1.0 / n] * n
    half_spread = [0.8 / half] * half + [0.2 / (n - half)] * (n - half)
    concentrated = [0.8] + [0.2 / (n - 1)] * (n - 1)

    return {
        "uniform": scales_for_target_shares(uniform),
        "half_spread": scales_for_target_shares(half_spread),
        "concentrated": scales_for_target_shares(concentrated),
    }


WORKLOAD_CASES = build_workload_cases()
SELECTED_CASES = [
    item.strip()
    for item in os.environ.get("INIT_CASES", ",".join(WORKLOAD_CASES.keys())).split(",")
    if item.strip()
]


def parse_centralized(text):
    for line in text.splitlines():
        if "总目标函数值" in line or "模型 objective" in line:
            return float(line.split(":")[-1].strip())
    raise RuntimeError("Failed to parse centralized objective:\n" + text[-2000:])


def demand_scale_json(scales):
    return json.dumps({str(i): float(scales[i]) for i in range(NUM_NODES)})


def cleanup_temp_outputs():
    for name in TEMP_OUTPUTS:
        path = BASE_DIR / name
        if path.exists():
            path.unlink()


def compute_centralized(scales):
    env = os.environ.copy()
    env.update(
        {
            "BASE_DIR": str(BASE_DIR),
            "TOPOLOGY": "full",
            "DEMAND_SCALE_JSON": demand_scale_json(scales),
            "ROUND_SCALED_DEMAND": COMMON_ENV["ROUND_SCALED_DEMAND"],
        }
    )
    result = subprocess.run(
        [PYTHON, str(CENTRALIZED_SCRIPT), DATASET],
        env=env,
        capture_output=True,
        text=True,
        timeout=RUN_TIMEOUT,
        check=False,
    )
    return parse_centralized(result.stdout + result.stderr)


def first_persistent_below_one(values):
    for idx in range(len(values)):
        tail = values[idx:]
        if tail.notna().all() and (tail <= 1.0).all():
            return int(idx + 1)
    return None


def run_case(case_name, scales):
    if MPIEXEC is None:
        raise RuntimeError("mpiexec was not found. Set MPIEXEC=/path/to/mpiexec.")

    print(f"\n[{case_name}] solving centralized baseline...", flush=True)
    centralized_base = compute_centralized(scales)
    print(f"[{case_name}] centralized objective = {centralized_base:.6f}", flush=True)
    env = os.environ.copy()
    env.update(COMMON_ENV)
    env["DEMAND_SCALE_JSON"] = demand_scale_json(scales)
    env["CENTRALIZED_BASE"] = str(centralized_base)

    started = time.time()
    cmd = [MPIEXEC, "-np", str(NUM_NODES), PYTHON, str(ASYNC_SCRIPT), DATASET]
    print(f"[{case_name}] running asynchronous ADMM with MAX_ITER={INIT_MAX_ITER}...", flush=True)
    try:
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=RUN_TIMEOUT)
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        captured = stdout + stderr
        raise RuntimeError(
            f"Run timed out for workload case {case_name} after {RUN_TIMEOUT}s:\n"
            + captured[-3000:]
        ) from exc

    metrics_path = BASE_DIR / "admm_async_auxu_metrics.csv"
    trace_path = BASE_DIR / "admm_trace_async.csv"
    if (
        not metrics_path.exists()
        or not trace_path.exists()
        or metrics_path.stat().st_mtime < started
        or trace_path.stat().st_mtime < started
    ):
        raise RuntimeError(
            f"Run failed for workload case {case_name}:\n"
            + (result.stdout + result.stderr)[-3000:]
        )

    metrics = pd.read_csv(metrics_path, encoding="utf-8-sig").iloc[0].to_dict()
    trace = pd.read_csv(trace_path, encoding="utf-8-sig")
    p_iter = first_persistent_below_one(trace["norm_dx_global"])
    d_iter = first_persistent_below_one(trace["norm_du_global"])
    both_iter = first_persistent_below_one(
        pd.concat([trace["norm_dx_global"], trace["norm_du_global"]], axis=1).max(axis=1)
    )

    row = {
        "case": case_name,
        "dataset": DATASET,
        "u_init_mode": "zero",
        "centralized_obj": centralized_base,
        "rel_gap_pct": metrics["rel_diff_to_centralized"] * 100.0,
        "wall_clock_total_s": metrics["wall_clock_total"],
        "avg_iter": metrics["avg_iter"],
        "avg_msg_send_cnt": metrics["avg_msg_send_cnt"],
        "iter_balance_pct": metrics["iter_balance_pct"],
        "persistent_primal_iter": p_iter,
        "persistent_dual_iter": d_iter,
        "persistent_both_iter": both_iter,
    }
    print(
        f"{case_name:>12s}: gap={row['rel_gap_pct']:.4f}%, "
        f"time={row['wall_clock_total_s']:.3f}s, both={both_iter}"
    )
    cleanup_temp_outputs()
    return row, trace


def plot_results(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogLocator, NullFormatter

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2,
        "legend.fontsize": 7.0,
        "axes.linewidth": 0.75,
        "lines.linewidth": 1.35,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    labels = {
        "uniform": "Uniform",
        "half_spread": "Half-spread",
        "concentrated": "Concentrated",
    }
    line_styles = ["-", "--", "-."]
    line_widths = [1.5, 1.3, 1.3]

    fig, axes = plt.subplots(2, 1, figsize=(3.55, 4.1), dpi=600)

    ax = axes[0]
    for idx, (row, trace) in enumerate(results):
        ax.plot(
            trace["iter"],
            np.clip(trace["norm_dx_global"], PLOT_Y_MIN, PLOT_Y_MAX),
            linestyle=line_styles[idx % len(line_styles)],
            linewidth=line_widths[idx % len(line_widths)],
            label=labels.get(row["case"], row["case"]),
        )
    ax.axhline(y=1.0, color="k", linestyle="-", linewidth=0.9, alpha=0.75)
    ax.set_yscale("log")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Normalized primal residual")
    ax.set_ylim(PLOT_Y_MIN, PLOT_Y_MAX)
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=6))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, which="major", linestyle="--", linewidth=0.38)
    ax.set_axisbelow(True)
    ax.text(0.02, 0.93, "(a)", transform=ax.transAxes, ha="left", va="top")

    ax = axes[1]
    for idx, (row, trace) in enumerate(results):
        ax.plot(
            trace["iter"],
            np.clip(trace["norm_du_global"], PLOT_Y_MIN, PLOT_Y_MAX),
            linestyle=line_styles[idx % len(line_styles)],
            linewidth=line_widths[idx % len(line_widths)],
            label=labels.get(row["case"], row["case"]),
        )
    ax.axhline(y=1.0, color="k", linestyle="-", linewidth=0.9, alpha=0.75)
    ax.set_yscale("log")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Normalized dual residual")
    ax.set_ylim(PLOT_Y_MIN, PLOT_Y_MAX)
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=6))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, which="major", linestyle="--", linewidth=0.38)
    ax.set_axisbelow(True)
    ax.text(0.02, 0.93, "(b)", transform=ax.transAxes, ha="left", va="top")

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.52, 1.0),
        ncol=len(results),
        frameon=True,
        framealpha=0.92,
        edgecolor="#BBBBBB",
        handlelength=2.2,
        columnspacing=0.8,
        handletextpad=0.4,
        borderpad=0.4,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93), pad=0.3, h_pad=0.7)
    fig.savefig(PLOT_PATH, dpi=600, bbox_inches="tight")
    plt.close(fig)


def main():
    unknown = [name for name in SELECTED_CASES if name not in WORKLOAD_CASES]
    if unknown:
        raise ValueError(f"Unknown INIT_CASES: {unknown}. Available: {list(WORKLOAD_CASES)}")

    cleanup_temp_outputs()
    results = [run_case(name, WORKLOAD_CASES[name]) for name in SELECTED_CASES]
    plot_results(results)
    print(f"\nSaved plot: {PLOT_PATH}")


if __name__ == "__main__":
    main()
