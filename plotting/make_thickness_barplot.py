import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

BASE = Path("/home/ubuntu/Downloads/mars_clean")

MODELS = [
    ("bm_github",  "outputs_bm"),
    ("bmw_github", "outputs_bmw"),
    ("HS",         "outputs_hsu"),
    ("HSh",        "outputs_hsu"),
    ("DEMV",       "outputs_demni"),
    ("DEMVH",      "outputs_demni"),
    ("kt",         "outputs_kt"),
    ("vrh",        "outputs_vrh"),
]

CASES = [
    ("A_constraints_away",  r'$V_p = 3.8$ km/s'),
    ("B_wright_inherited",  r'$V_p = 4.1$ km/s'),
    ("C_insight_marsquake", r'$V_p = 4.7$ km/s'),
]

theories = list(range(1, len(MODELS) + 1))
n_theories = len(theories)
n_cases = len(CASES)

# Compute mode/median/mean from .npy files
mode   = np.zeros((n_theories, n_cases))
median = np.zeros((n_theories, n_cases))
mean   = np.zeros((n_theories, n_cases))

for i, (model_key, out_sub) in enumerate(MODELS):
    for j, (case_suffix, _) in enumerate(CASES):
        npy = BASE / model_key / out_sub / f"thickness_samples_{case_suffix}.npy"
        s = np.load(npy) / 1000.0  # m -> km
        counts, edges = np.histogram(s, bins=200)
        mode_bin = np.argmax(counts)
        mode[i, j]   = (edges[mode_bin] + edges[mode_bin + 1]) / 2
        median[i, j] = np.median(s)
        mean[i, j]   = np.mean(s)

vp_labels = [label for _, label in CASES]

# Shared x-axis
xmax = max(mode.max(), median.max(), mean.max()) * 1.05

y = np.arange(n_theories)
height = 0.25

fig, axes = plt.subplots(1, 3, figsize=(14, 8), sharex=True)

# Colorblind-friendly palette (blue / orange / grey)
colors = ['#0072B2', '#E69F00', '#999999']

for col, ax in enumerate(axes):
    ax.barh(y + height, mode[:, col], height, color=colors[0], edgecolor='black', linewidth=0.5)
    ax.barh(y,          median[:, col], height, color=colors[1], edgecolor='black', linewidth=0.5)
    ax.barh(y - height, mean[:, col], height, color=colors[2], edgecolor='black', linewidth=0.5)

    ax.set_xticks(np.arange(0, 4.0, 0.5))
    ax.set_yticks(y)
    ax.set_yticklabels(theories, fontsize=20)
    ax.set_title(vp_labels[col], fontsize=22)
    ax.set_xlim(0, xmax)
    ax.tick_params(axis='both', which='major', labelsize=20, length=8, width=1.5)
    ax.invert_yaxis()

axes[0].set_ylabel('Theory Number', fontsize=22)
axes[1].set_xlabel('Water-Layer Thickness (km)', fontsize=22)

# Mark the mode-based estimate from peak porosity & saturation (Wright et al.)
WRIGHT_MODE = 1.248
ax1 = axes[1]
ax1.plot(WRIGHT_MODE, y[0] + height, marker='*', color='red', markersize=18,
         markeredgecolor='black', markeredgewidth=0.8, zorder=10,
         label='Peak-based mode')

# Build legend manually
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
legend_elements = [
    Patch(facecolor=colors[0], edgecolor='black', linewidth=0.5, label='Mode'),
    Patch(facecolor=colors[1], edgecolor='black', linewidth=0.5, label='Median'),
    Patch(facecolor=colors[2], edgecolor='black', linewidth=0.5, label='Mean'),
    Line2D([0], [0], marker='*', color='w', markerfacecolor='red',
           markeredgecolor='black', markeredgewidth=0.8, markersize=18,
           label='Peak-based mode'),
]
axes[1].legend(handles=legend_elements, fontsize=14, loc='center right')

plt.tight_layout()

# Red box around theory 1, Vp = 4.1 (Wright et al. reproduction)
ax1 = axes[1]
theory1_y = y[0]
pad_y = 0.2
bar_max = max(mode[0, 1], median[0, 1], mean[0, 1])
x0, x1 = 0.02, WRIGHT_MODE + 0.25
y0 = theory1_y - height - height / 2 - pad_y
y1 = theory1_y + height + height / 2 + pad_y
inv = fig.transFigure.inverted()
p0 = inv.transform(ax1.transData.transform((x0, y0)))
p1 = inv.transform(ax1.transData.transform((x1, y1)))
fig_rect = mpatches.FancyBboxPatch(
    (min(p0[0], p1[0]), min(p0[1], p1[1])),
    abs(p1[0] - p0[0]), abs(p1[1] - p0[1]),
    boxstyle="round,pad=0.005",
    linewidth=3, edgecolor='red', facecolor='none',
    transform=fig.transFigure, zorder=100)
fig.patches.append(fig_rect)

fig.savefig("thickness_barplot.png", dpi=300, bbox_inches='tight')
fig.savefig("thickness_barplot.pdf", bbox_inches='tight')
plt.close(fig)
print(f"Saved thickness_barplot.png and .pdf (x-axis: 0 to {xmax:.2f} km)")
