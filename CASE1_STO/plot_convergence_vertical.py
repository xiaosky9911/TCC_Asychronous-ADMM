from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter

base_dir = Path(".")
csv_files = [
    base_dir / "admm_trace_sync.csv",
    base_dir / "admm_trace_async_undelay.csv",
    base_dir / "admm_trace_async_delay.csv",
    base_dir / "admm_trace_async.csv",
]

labels = ["Sync-ADMM", "Naive Async (ASP)", "BD-Async (SSP)", "MSMD-Async"]
centralized_obj = 1182823039.614968

traces = []
for path, label in zip(csv_files, labels):
    df = pd.read_csv(path).sort_values("iter").reset_index(drop=True)
    df["gap_pct"] = np.abs(df["obj_global"] - centralized_obj) / abs(centralized_obj) * 100
    traces.append((label, df))

def compress_duplicate_x(x, y):
    tmp = pd.DataFrame({"x": np.asarray(x, dtype=float), "y": np.asarray(y, dtype=float)})
    tmp = tmp.groupby("x", as_index=False, sort=True).tail(1)
    tmp = tmp.sort_values("x", kind="mergesort")
    return tmp["x"].to_numpy(), tmp["y"].to_numpy()

def pad_to_target(x, y, target):
    if len(x) > 0 and target > x[-1]:
        x = np.r_[x, target]
        y = np.r_[y, y[-1]]
    return x, y

def get_series(df, x_mode, target=None):
    if x_mode == "time":
        x, y = compress_duplicate_x(df["wall_time_global"], df["gap_pct"])
    else:
        x = df["iter"].to_numpy(dtype=float)
        y = df["gap_pct"].to_numpy(dtype=float)
    if target is not None:
        x, y = pad_to_target(x, y, target)
    return x, y

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

line_styles = ["-", "--", "-.", ":"]
line_widths = [1.25, 1.25, 1.25, 1.85]

target_time = max(compress_duplicate_x(df["wall_time_global"], df["gap_pct"])[0][-1] for _, df in traces)
target_iter = max(df["iter"].iloc[-1] for _, df in traces)

fig, axes = plt.subplots(2, 1, figsize=(3.55, 4.1), dpi=600, sharey=True)

for ax, x_mode, xlabel, panel_label in [
    (axes[0], "iter", "Iteration",           "(a)"),
    (axes[1], "time", "Wall-clock time (s)", "(b)"),
]:
    target_x = target_time if x_mode == "time" else target_iter
    for i, (label, df) in enumerate(traces):
        x, y = get_series(df, x_mode, target_x)
        ax.plot(x, y, linestyle=line_styles[i], linewidth=line_widths[i], label=label)

    ax.set_yscale("log")
    ax.set_ylim(8e-4, 3e1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Relative gap (%)")
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=6))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, which="major", linestyle="--", linewidth=0.38)
    ax.set_axisbelow(True)
    ax.text(0.02, 0.93, panel_label, transform=ax.transAxes, ha="left", va="top", fontsize=8)

handles, legend_labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    legend_labels,
    loc="upper center",
    bbox_to_anchor=(0.52, 1.0),
    ncol=2,
    frameon=True,
    framealpha=0.92,
    edgecolor="#BBBBBB",
    handlelength=2.2,
    columnspacing=0.8,
    handletextpad=0.4,
    borderpad=0.4,
    labelspacing=0.3,
)

fig.tight_layout(rect=(0, 0, 1, 0.905), pad=0.3, h_pad=0.7)

# fig.savefig("convergence_gap_two_panel_vertical.pdf", bbox_inches="tight")
# fig.savefig("convergence_gap_two_panel_vertical.eps", bbox_inches="tight")
fig.savefig("convergence_gap_sto.png", dpi=600, bbox_inches="tight")
# plt.show()
