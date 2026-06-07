import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]

# (model_dir, output_subdir, saturation_col_prefix_or_None, saturation_col_fallback, label)
MODELS = [
    ("models/1_SCM_Voigt",  "outputs_bm",    "demni",  None, "1: Berryman + Voigt"),
    ("models/2_SCM_Hill", "outputs_bmw",   "demni",  None, "2: Berryman + Hill"),
    ("models/3_HS_Voigt",         "outputs_hsu",   "hs",     None, "3: HS + Voigt"),
    ("models/4_HS_Hill",        "outputs_hsu",   None,     2,    "4: HS + Hill"),
    ("models/5_DEM_Voigt",       "outputs_demni", "demni",  None, "5: DEM + Voigt"),
    ("models/6_DEM_Hill",      "outputs_demni", "demni",  None, "6: DEM + Hill"),
    ("models/7_KT",         "outputs_kt",    "models/7_KT",     None, "7: Kuster\u2013Toks\u00f6z"),
    ("models/8_VRH",        "outputs_vrh",   "models/8_VRH",    None, "8: Voigt\u2013Reuss\u2013Hill"),
]

CASES = [
    ("A_constraints_away",  r"$V_p=3.8$ km/s", "#0072B2"),   # blue
    ("B_wright_inherited",  r"$V_p=4.1$ km/s", "#E69F00"),   # orange
    ("C_insight_marsquake", r"$V_p=4.7$ km/s", "#999999"),   # grey
]

XLIM = (0, 1.0)
X_EVAL = np.linspace(XLIM[0], XLIM[1], 500)
rng = np.random.default_rng(42)

# Precompute all KDEs and find per-row max for scaling
all_densities = {}
row_max = {}
for idx, (model_key, out_sub, prefix, fallback_col, _) in enumerate(MODELS):
    rmax = 0
    for case_suffix, _, _ in CASES:
        outdir = BASE / model_key / out_sub
        # Get saturation column index
        if prefix is not None:
            scol = int(np.load(outdir / f"saturation_{prefix}_{case_suffix}.npy"))
        else:
            scol = fallback_col
        samples = np.load(outdir / f"samples_{case_suffix}.npy")
        saturation = samples[:, scol]
        sub = rng.choice(saturation, size=min(50000, len(saturation)), replace=False)
        kde = gaussian_kde(sub, bw_method=0.05)
        density = kde(X_EVAL)
        all_densities[(idx, case_suffix)] = density
        rmax = max(rmax, density.max())
    row_max[idx] = rmax

# Ridgeline parameters
n_rows = len(MODELS)
row_height = 1.0

# Single common vertical scale shared by every theory row
global_max = max(row_max.values())

fig, ax = plt.subplots(figsize=(10, 10))

for idx in range(n_rows - 1, -1, -1):  # draw bottom rows first
    baseline = idx * row_height
    theory_idx = n_rows - 1 - idx  # theory 1 at top

    scale = row_height * 0.85 / global_max

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
ax.set_xlabel("Water Saturation", fontsize=16)
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

# Single shared density axis on the right: all rows use one common vertical
# scale, so one calibrated ridge-height bar (0 -> global_max) serves every row.
from matplotlib.ticker import MaxNLocator
pos = ax.get_position()
ylim = ax.get_ylim()
data_range = ylim[1] - ylim[0]
bar_fig_height = (row_height * 0.85) / data_range * (pos.y1 - pos.y0)
fig_yc = (pos.y0 + pos.y1) / 2
ax_r = fig.add_axes([pos.x1 + 0.005, fig_yc - bar_fig_height / 2, 0.015, bar_fig_height])
ax_r.set_ylim(0, global_max)
ax_r.set_xticks([])
ax_r.yaxis.tick_right()
ax_r.yaxis.set_major_locator(MaxNLocator(nbins=4, prune='both'))
ax_r.tick_params(axis='y', labelsize=8, length=3, width=0.8)
for s in ('top', 'bottom', 'left'):
    ax_r.spines[s].set_visible(False)

# Right-side axis label
fig.text(pos.x1 + 0.06, (pos.y0 + pos.y1) / 2, "Probability Density",
         fontsize=16, rotation=-90, va='center', ha='left')

fig.savefig("saturation_ridgeline.png", dpi=300, bbox_inches='tight')
fig.savefig("saturation_ridgeline.pdf", bbox_inches='tight')
plt.close(fig)
print("Saved saturation_ridgeline.png and .pdf")
