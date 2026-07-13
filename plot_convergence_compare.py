# -*- coding: utf-8 -*-
"""
绘制同步/异步收敛对比图。

默认画目标函数值 vs 墙钟时间；如果输入 CSV 没有 wall_time_global，则自动退回到迭代次数。

默认输入:
    admm_trace_sync.csv
    admm_trace_async_undelay.csv
    admm_trace_async_delay.csv
    admm_trace_async.csv

默认输出: convergence_compare.png
"""

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import ScalarFormatter

CENTRALIZED_OBJECTIVE = 1182823039.614968

PALETTE = ["#003F5C", "#2A9D8F", "#6A4C93", "#D62828"]
LINESTYLES = ["-", "-", "--", "-."]
TITLE_MAP = {
    "wall_time": "Convergence Comparison vs. Wall-clock Time",
    "iter": "Convergence Comparison vs. Iterations",
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot sync/async convergence comparison.")
    parser.add_argument(
        "csv_paths",
        nargs="*",
        help="Optional four trace CSV files: sync + three async variants. Defaults to local files if omitted.",
    )
    parser.add_argument(
        "--labels",
        nargs=4,
        default=[
            "Sync-ADMM",
            "Naive Async (ASP)",
            "BD-Async (SSP)",
            "MSMD-Async",
        ],
        help="Legend labels for the four curves.",
    )
    parser.add_argument(
        "--out",
        default=os.path.join(os.getcwd(), "convergence_compare.png"),
        help="Output image path.",
    )
    parser.add_argument(
        "--x-axis",
        choices=["wall_time", "iter"],
        default="wall_time",
        help="Prefer wall time or iteration count on the x-axis.",
    )
    return parser


def _load_trace(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到轨迹文件: {path}")
    df = pd.read_csv(path)
    if "obj_global" not in df.columns:
        raise ValueError(f"轨迹文件缺少 obj_global 字段: {path}")
    if "iter" not in df.columns:
        raise ValueError(f"轨迹文件缺少 iter 字段: {path}")
    return df


def _select_x(df: pd.DataFrame, preferred: str) -> tuple[str, pd.Series]:
    if preferred == "wall_time" and "wall_time_global" in df.columns:
        return "wall_time_global", df["wall_time_global"]
    return "iter", df["iter"]


def _select_x_for_plot(df: pd.DataFrame, x_axis: str) -> tuple[str, pd.Series]:
    if x_axis == "wall_time" and "wall_time_global" in df.columns:
        return "wall_time_global", df["wall_time_global"]
    return "iter", df["iter"]


def _pad_trace(x_values: pd.Series, y_values: pd.Series, target_x: float) -> tuple[pd.Series, pd.Series]:
    """Extend a trace to target_x with a flat tail at the final objective value."""
    if len(x_values) == 0:
        return x_values, y_values

    last_x = float(x_values.iloc[-1])
    last_y = float(y_values.iloc[-1])

    if target_x <= last_x:
        return x_values, y_values

    extended_x = pd.concat([x_values, pd.Series([target_x], dtype=float)], ignore_index=True)
    extended_y = pd.concat([y_values, pd.Series([last_y], dtype=float)], ignore_index=True)
    return extended_x, extended_y


def _compress_duplicate_x(x_values: pd.Series, y_values: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Keep only the last y value for each duplicated x so the curve stays monotone on the x-axis."""
    if len(x_values) == 0:
        return x_values, y_values

    df = pd.DataFrame({"x": x_values.astype(float).to_numpy(), "y": y_values.astype(float).to_numpy()})
    df = df.groupby("x", as_index=False, sort=True).tail(1).sort_values("x", kind="mergesort")
    return pd.Series(df["x"].to_numpy(), dtype=float), pd.Series(df["y"].to_numpy(), dtype=float)


def _plot_traces(traces: list[tuple[str, pd.DataFrame]], x_axis: str, out_path: str) -> None:
    """Draw one comparison figure for the requested x-axis type."""
    fig, ax = plt.subplots(figsize=(10.8, 7.2))
    x_label = "Wall-clock Time (s)" if x_axis == "wall_time" else "Iteration"

    x_ends = []
    for _, df in traces:
        x_name, x_values = _select_x_for_plot(df, x_axis)
        if len(x_values) > 0:
            x_ends.append(float(x_values.iloc[-1]))
    target_x = max(x_ends) if len(x_ends) > 0 else 0.0

    for idx, (label, df) in enumerate(traces):
        x_name, plot_x = _select_x_for_plot(df, x_axis)
        plot_y = df["obj_global"]

        if x_axis == "wall_time" and x_name == "wall_time_global":
            plot_x, plot_y = _compress_duplicate_x(plot_x, plot_y)
            plot_x, plot_y = _pad_trace(plot_x, plot_y, target_x)
        elif x_axis == "iter":
            plot_x, plot_y = _pad_trace(plot_x, plot_y, target_x)

        ax.plot(
            plot_x,
            plot_y,
            color=PALETTE[idx % len(PALETTE)],
            linestyle=LINESTYLES[idx % len(LINESTYLES)],
            linewidth=3.0 if label == "Proposed" else 2.2,
            zorder=4 if label == "Proposed" else 3,
            label=label,
        )

    ax.axhline(
        y=CENTRALIZED_OBJECTIVE,
        color="#2CA02C",
        linestyle=":",
        linewidth=2.0,
        label="Centralized",
    )

    ax.set_xlabel(x_label)
    ax.set_ylabel("Global Objective Value")
    ax.set_axisbelow(True)
    ax.grid(True, which="major", linestyle="--", linewidth=0.8, alpha=0.55)
    ax.minorticks_on()
    ax.grid(True, which="minor", linestyle=":", linewidth=0.55, alpha=0.28)

    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 3))
    ax.yaxis.set_major_formatter(formatter)

    ax.set_title(TITLE_MAP.get(x_axis, "Convergence Comparison"), pad=12)
    ax.legend(loc="best", frameon=True, edgecolor="black")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"绘图完成，已保存: {out_path}")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_csvs = [
        os.path.join(base_dir, "admm_trace_sync.csv"),
        os.path.join(base_dir, "admm_trace_async_undelay.csv"),
        os.path.join(base_dir, "admm_trace_async_delay.csv"),
        os.path.join(base_dir, "admm_trace_async.csv"),
    ]

    csv_paths = args.csv_paths if len(args.csv_paths) == 4 else default_csvs

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["font.size"] = 15
    plt.rcParams["axes.labelsize"] = 16
    plt.rcParams["legend.fontsize"] = 12
    plt.rcParams["axes.linewidth"] = 1.3
    plt.rcParams["savefig.facecolor"] = "white"
    plt.rcParams["savefig.edgecolor"] = "white"

    traces = []
    for path, label in zip(csv_paths, args.labels):
        df = _load_trace(path)
        traces.append((label, df))

    wall_out = args.out
    iter_out = os.path.join(os.path.dirname(args.out), "convergence_compare_iter.png")

    _plot_traces(traces, "wall_time", wall_out)
    _plot_traces(traces, "iter", iter_out)


if __name__ == "__main__":
    main()
