import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# Data from Table IX
# =========================
systems = np.array([8, 16, 32])

sync_time = np.array([3.72, 18.39, 63.56])
async_time = np.array([2.70, 6.67, 14.40])

sync_iter = np.array([84, 413, 1334])
async_iter = np.array([79, 160, 301])

sync_msg = np.array([588, 6195, 41354])
async_msg = np.array([573, 2450, 9449])

# =========================
# Reduction over Sync. (%)
# =========================
time_red = (1 - async_time / sync_time) * 100
iter_red = (1 - async_iter / sync_iter) * 100
msg_red = (1 - async_msg / sync_msg) * 100

metrics = {
    "Wall-clock time": time_red,
    "Iterations": iter_red,
    "Messages": msg_red,
}

# =========================
# Figure style
# =========================
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.2,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

x = np.arange(len(systems))
width = 0.23

fig, ax = plt.subplots(figsize=(3.45, 2.25), dpi=600)

hatches = ["", "///", "\\\\\\"]
colors = ["#FFE599", "#A8D5A2", "#A8C8E8"]   # 浅黄 / 浅绿 / 浅蓝

for idx, (label, values) in enumerate(metrics.items()):
    bars = ax.bar(
        x + (idx - 1) * width,
        values,
        width,
        label=label,
        color=colors[idx],
        edgecolor="black",
        linewidth=0.7,
        hatch=hatches[idx],
    )

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.2,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=6.5,
        )

# =========================
# Axis settings
# =========================
ax.set_xticks(x)
ax.set_xticklabels([str(s) for s in systems])
ax.set_xlabel("Number of regions")
ax.set_ylabel("Reduction over Sync. (%)")

ax.set_ylim(0, 88)
ax.set_yticks(np.arange(0, 91, 20))

ax.grid(axis="y", linestyle="-", linewidth=0.4, alpha=0.35)
ax.set_axisbelow(True)

ax.legend(
    frameon=False,
    loc="upper left",
    handlelength=1.5,
)

fig.tight_layout(pad=0.35)

# =========================
# Save figure
# =========================
out_dir = Path(".")
fig.savefig(out_dir / "scalability_reduction_async_sync.png", dpi=600, bbox_inches="tight")

plt.show()