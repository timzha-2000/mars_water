import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from pathlib import Path

BASE = Path("/home/ubuntu/Downloads/mars_clean")

# (model_dir, output_subdir, porosity_col_prefix_or_None, porosity_col_fallback, label)
MODELS = [
    ("bm_github",  "outputs_bm",    "demni",  None, "1: Berryman + Voigt"),
    ("bmw_github", "outputs_bmw",   "demni",  None, "2: Berryman + Hill"),
    ("HS",         "outputs_hsu",   "hs",     None, "3: HS + Voigt"),
    ("HSh",        "outputs_hsu",   None,     1,    "4: HS + Hill"),
    ("DEMV",       "outputs_demni", "demni",  None, "5: DEM + Voigt"),
    ("DEMVH",      "outputs_demni", "demni",  None, "6: DEM + Hill"),
    ("kt",         "outputs_kt",    "kt",     None, "7: Kuster\u2013Toks\u00f6z"),
    ("vrh",        "outputs_vrh",   "vrh",    None, "8: Voigt\u2013Reuss\u2013Hill"),
]

CASES = [
    ("A_constraints_away",  r"$V_p=3.8$ km/s", "#0072B2"),   # blue
    ("B_wright_inherited",  r"$V_p=4.1$ km/s", "#E69F00"),   # orange
    ("C_insight_marsquake", r"$V_p=4.7$ km/s", "#999999"),   # grey
]

XLIM = (0, 0.5)
X_EVAL = np.linspace(XLIM[0], XLIM[1], 500)
rng = np.random.default_rng(42)

# Precompute all KDEs and find per-row max for scaling
all_densities = {}
row_max = {}
for idx, (model_key, out_sub, prefix, fallback_col, _) in enumerate(MODELS):
    rmax = 0
    for case_suffix, _, _ in CASES:
        outdir = BASE / model_key / out_sub
        # Get porosity column index
        if prefix is not None:
            pcol = int(np.load(outdir / f"porosity_{prefix}_{case_suffix}.npy"))
        else:
            pcol = fallback_col
        samples = np.load(outdir / f"samples_{case_suffix}.npy")
        porosity = samples[:, pcol]
        sub = rng.choice(porosity, size=min(50000, len(porosity)), replace=False)
        kde = gaussian_kde(sub, bw_method=0.05)
        density = kde(X_EVAL)
        all_densities[(idx, case_suffix)] = density
        rmax = max(rmax, density.max())
    row_max[idx] = rmax

# Ridgeline parameters
n_rows = len(MODELS)
row_height = 1.0

fig, ax = plt.subplots(figsize=(10, 10))

for idx in range(n_rows - 1, -1, -1):  # draw bottom rows first
    baseline = idx * row_height
    theory_idx = n_rows - 1 - idx  # theory 1 at top

    scale = row_height * 0.85 / row_max[theory_idx]

    for case_suffix, case_label, color in reversed(CASES):
        density = all_densities[(theory_idx, case_suffix)]
        y_vals = baseline + density * scale

        ax.fill_between(X_EVAL, baseline, y_vals, alpha=0.25, color=color)
        ax.plot(X_EVAL, y_vals, color=color, linewidth=1.5,
                label=case_label if idx == n_rows - 1 else None)

    ax.axhline(baseline, color='grey', linewidth=0.4, zorder=0)

# Y-axis: theory labels at each baseline
ax.set_yticks([i * row_height for i in range(n_rows)])
ax.set_yticklabels([MODELS[n_rows - 1 - i][4] for i in range(n_rows)], fontsize=13)

ax.set_xlim(*XLIM)
ax.set_ylim(-0.1, n_rows * row_height + 0.1)
ax.set_xlabel("Porosity", fontsize=16)
ax.set_ylabel("")
ax.tick_params(axis='x', which='major', labelsize=14, length=6, width=1.2)
ax.tick_params(axis='y', which='major', length=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)

# Legend
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc='upper right', fontsize=14, frameon=True)

plt.tight_layout()

# Add a right-side vertical axis for each row showing actual density scale
from matplotlib.ticker import MaxNLocator
pos = ax.get_position()
ylim = ax.get_ylim()
data_range = ylim[1] - ylim[0]
for idx in range(n_rows):
    theory_idx = n_rows - 1 - idx
    baseline = idx * row_height
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

fig.savefig("porosity_ridgeline.png", dpi=300, bbox_inches='tight')
fig.savefig("porosity_ridgeline.pdf", bbox_inches='tight')
plt.close(fig)
print("Saved porosity_ridgeline.png and .pdf")
