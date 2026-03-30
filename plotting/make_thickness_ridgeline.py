import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from pathlib import Path

BASE = Path("/home/ubuntu/Downloads/mars_clean")

MODELS = [
    ("bm_github",  "outputs_bm",    "1: Berryman + Voigt"),
    ("bmw_github", "outputs_bmw",   "2: Berryman + Hill"),
    ("HS",         "outputs_hsu",   "3: HS + Voigt"),
    ("HSh",        "outputs_hsu",   "4: HS + Hill"),
    ("DEMV",       "outputs_demni", "5: DEM + Voigt"),
    ("DEMVH",      "outputs_demni", "6: DEM + Hill"),
    ("kt",         "outputs_kt",    "7: Kuster\u2013Toks\u00f6z"),
    ("vrh",        "outputs_vrh",   "8: Voigt\u2013Reuss\u2013Hill"),
]

CASES = [
    ("A_constraints_away",  r"$V_p=3.8$ km/s", "#0072B2"),   # blue
    ("B_wright_inherited",  r"$V_p=4.1$ km/s", "#E69F00"),   # orange
    ("C_insight_marsquake", r"$V_p=4.7$ km/s", "#999999"),   # grey
]

XLIM = (0, 4.0)
X_EVAL = np.linspace(XLIM[0], XLIM[1], 500)
rng = np.random.default_rng(42)

# Precompute all KDEs and find per-row max for scaling
all_densities = {}
row_max = {}
for idx, (model_key, out_sub, _) in enumerate(MODELS):
    rmax = 0
    for case_suffix, _, _ in CASES:
        npy = BASE / model_key / out_sub / f"thickness_samples_{case_suffix}.npy"
        samples_km = np.load(npy) / 1000.0
        sub = rng.choice(samples_km, size=min(50000, len(samples_km)), replace=False)
        kde = gaussian_kde(sub, bw_method=0.05)
        density = kde(X_EVAL)
        all_densities[(idx, case_suffix)] = density
        rmax = max(rmax, density.max())
    row_max[idx] = rmax

# Ridgeline parameters
n_rows = len(MODELS)
row_height = 1.0        # vertical spacing between rows

fig, ax = plt.subplots(figsize=(10, 10))

for idx in range(n_rows - 1, -1, -1):  # draw bottom rows first
    baseline = idx * row_height
    model_key, out_sub, label = MODELS[n_rows - 1 - idx]  # theory 1 at top

    # Per-row scaling so each ridge fills its row
    theory_idx = n_rows - 1 - idx
    scale = row_height * 0.85 / row_max[theory_idx]

    # Draw in reverse order so first case is on top
    for case_suffix, case_label, color in reversed(CASES):
        density = all_densities[(theory_idx, case_suffix)]
        y_vals = baseline + density * scale

        ax.fill_between(X_EVAL, baseline, y_vals, alpha=0.25, color=color)
        ax.plot(X_EVAL, y_vals, color=color, linewidth=1.5, label=case_label if idx == n_rows - 1 else None)

    # Baseline
    ax.axhline(baseline, color='grey', linewidth=0.4, zorder=0)

# Y-axis: theory labels at each baseline
ax.set_yticks([i * row_height for i in range(n_rows)])
ax.set_yticklabels([MODELS[n_rows - 1 - i][2] for i in range(n_rows)], fontsize=13)

ax.set_xlim(*XLIM)
ax.set_ylim(-0.1, n_rows * row_height + 0.1)
ax.set_xlabel("Water-Layer Thickness (km)", fontsize=16)
ax.set_ylabel("")
ax.tick_params(axis='x', which='major', labelsize=14, length=6, width=1.2)
ax.tick_params(axis='y', which='major', length=0)  # hide y ticks
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)

# Wright et al. peak-based mode estimate on theory 1, Vp=4.1 row
WRIGHT_MODE = 1.248
theory1_baseline = (n_rows - 1) * row_height  # theory 1 is at the top
ax.plot(WRIGHT_MODE, theory1_baseline, marker='*', color='red', markersize=18,
        markeredgecolor='black', markeredgewidth=0.8, zorder=10)

# Legend
from matplotlib.lines import Line2D
handles, labels = ax.get_legend_handles_labels()
handles.append(Line2D([0], [0], marker='*', color='w', markerfacecolor='red',
                       markeredgecolor='black', markeredgewidth=0.8, markersize=18,
                       label='Peak-based mode'))
labels.append('Peak-based mode')
ax.legend(handles, labels, loc='center right', fontsize=14, frameon=True)

plt.tight_layout()

# Add a right-side vertical axis for each row showing actual density scale
from matplotlib.ticker import MaxNLocator
pos = ax.get_position()
ylim = ax.get_ylim()
data_range = ylim[1] - ylim[0]
for idx in range(n_rows):
    theory_idx = n_rows - 1 - idx
    baseline = idx * row_height
    # Convert data coords to figure coords
    fig_y0 = pos.y0 + (baseline - ylim[0]) / data_range * (pos.y1 - pos.y0)
    fig_y1 = pos.y0 + (baseline + row_height - ylim[0]) / data_range * (pos.y1 - pos.y0)
    gap = (fig_y1 - fig_y0) * 0.08
    fig_y0 += gap
    fig_y1 -= gap
    ax_r = fig.add_axes([pos.x1, fig_y0, 0.015, fig_y1 - fig_y0])
    ax_r.set_ylim(0, row_max[theory_idx])
    ax_r.set_xticks([])
    ax_r.yaxis.tick_right()
    ax_r.yaxis.set_major_locator(MaxNLocator(nbins=3, prune='both'))
    ax_r.tick_params(axis='y', labelsize=7, length=3, width=0.8)
    ax_r.spines['top'].set_visible(False)
    ax_r.spines['bottom'].set_visible(False)
    ax_r.spines['left'].set_visible(False)

# Right-side axis label
fig.text(pos.x1 + 0.06, (pos.y0 + pos.y1) / 2, "Probability Density",
         fontsize=16, rotation=-90, va='center', ha='left')

fig.savefig("thickness_ridgeline.png", dpi=300, bbox_inches='tight')
fig.savefig("thickness_ridgeline.pdf", bbox_inches='tight')
plt.close(fig)
print("Saved thickness_ridgeline.png and .pdf")
