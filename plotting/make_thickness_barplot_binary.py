"""Binary-saturation analogue of make_thickness_barplot.py.

Each theory x case bar group shows mode/median/mean of the binary thickness
MIXTURE posterior: with prob P(wet) draw from the wet-run thickness samples,
with prob P(dry)=1-P(wet) thickness is 0. P(wet)=Z_wet/(Z_wet+Z_dry) from the
prior-MC evidence logs written by binary_sat/.
"""
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Patch

BASE = Path(__file__).resolve().parents[1]

# Same theories/labels as the continuous barplot
MODELS = [
    ("models/1_SCM_Voigt",  "1: Berryman + Voigt"),
    ("models/2_SCM_Hill", "2: Berryman + Hill"),
    ("models/3_HS_Voigt",         "3: HS + Voigt"),
    ("models/4_HS_Hill",        "4: HS + Hill"),
    ("models/5_DEM_Voigt",       "5: DEM + Voigt"),
    ("models/6_DEM_Hill",      "6: DEM + Hill"),
    ("models/7_KT",         "7: Kuster–Toksöz"),
    ("models/8_VRH",        "8: Voigt–Reuss–Hill"),
]
CASES = [
    ("A_constraints_away",  r'$V_p = 3.8$ km/s'),
    ("B_wright_inherited",  r'$V_p = 4.1$ km/s'),
    ("C_insight_marsquake", r'$V_p = 4.7$ km/s'),
]

BURN = 0.20   # drop first 20% of wet-run samples (matches make_binary_table)
NMIX = 200000 # mixture sample size

def read_logZ(path):
    try:
        return float(open(path).read())
    except Exception:
        return float("-inf")

def p_wet(zw, zd):
    if zw == float("-inf") and zd == float("-inf"):
        return float("nan")
    if zd == float("-inf"):
        return 1.0
    if zw == float("-inf"):
        return 0.0
    return 1.0 / (1.0 + math.exp(zd - zw))

def mixture_stats(model_key, case):
    zw = read_logZ(BASE / model_key / "outputs_binary_wet" / f"logZ_{case}.txt")
    zd = read_logZ(BASE / model_key / "outputs_binary_dry" / f"logZ_{case}.txt")
    pw = p_wet(zw, zd)

    wet = np.load(BASE / model_key / "outputs_binary_wet" / f"thickness_samples_{case}.npy") / 1000.0
    wet = wet[int(BURN * len(wet)):]

    rng = np.random.default_rng(0)
    pw_eff = 0.0 if math.isnan(pw) else pw
    n_wet = int(round(pw_eff * NMIX))
    mix = np.concatenate([
        rng.choice(wet, size=n_wet, replace=True) if n_wet > 0 else np.array([]),
        np.zeros(NMIX - n_wet),
    ])
    counts, edges = np.histogram(mix, bins=200)
    mode_bin = int(np.argmax(counts))
    mode = 0.5 * (edges[mode_bin] + edges[mode_bin + 1])
    return mode, float(np.median(mix)), float(mix.mean()), pw

n_theories = len(MODELS)
n_cases = len(CASES)

mode_a   = np.zeros((n_theories, n_cases))
median_a = np.zeros((n_theories, n_cases))
mean_a   = np.zeros((n_theories, n_cases))
pwet_a   = np.zeros((n_theories, n_cases))

for i, (model_key, _) in enumerate(MODELS):
    for j, (case_suffix, _) in enumerate(CASES):
        m, med, mn, pw = mixture_stats(model_key, case_suffix)
        mode_a[i, j], median_a[i, j], mean_a[i, j], pwet_a[i, j] = m, med, mn, pw

theory_labels = [name for _, name in MODELS]
vp_labels     = [label for _, label in CASES]

xmax = max(mode_a.max(), median_a.max(), mean_a.max()) * 1.05

y = np.arange(n_theories)
height = 0.25

fig, axes = plt.subplots(1, 3, figsize=(14, 8), sharex=True)
colors = ['#0072B2', '#E69F00', '#999999']  # mode / median / mean

for col, ax in enumerate(axes):
    ax.barh(y + height, mode_a[:, col],   height, color=colors[0], edgecolor='black', linewidth=0.5)
    ax.barh(y,          median_a[:, col], height, color=colors[1], edgecolor='black', linewidth=0.5)
    ax.barh(y - height, mean_a[:, col],   height, color=colors[2], edgecolor='black', linewidth=0.5)

    # Per-row P(wet) annotation just inside the right edge
    for ti in range(n_theories):
        ax.text(xmax * 0.98, y[ti], f"P(wet)={pwet_a[ti, col]:.2f}",
                ha='right', va='center', fontsize=10, color='black')

    ax.set_yticks(y)
    if col == 0:
        ax.set_yticklabels(theory_labels, fontsize=13)
    else:
        ax.set_yticklabels([])
    ax.set_title(vp_labels[col], fontsize=22)
    ax.set_xlim(0, xmax)
    ax.tick_params(axis='both', which='major', labelsize=20, length=8, width=1.5)
    ax.invert_yaxis()

axes[1].set_xlabel('Binary water-layer thickness (km)', fontsize=22)

legend_elements = [
    Patch(facecolor=colors[0], edgecolor='black', linewidth=0.5, label='Mode'),
    Patch(facecolor=colors[1], edgecolor='black', linewidth=0.5, label='Median'),
    Patch(facecolor=colors[2], edgecolor='black', linewidth=0.5, label='Mean'),
]
axes[1].legend(handles=legend_elements, fontsize=14, loc='center right')

plt.tight_layout()

# Red box around Theory 1 (SCM+Voigt) at Vp=4.1: Wright et al.'s reference combo
ax1 = axes[1]
theory1_y = y[0]
pad_y = 0.2
x0, x1 = 0.02, xmax * 0.62
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

out = BASE / "thickness_barplot_binary.png"
fig.savefig(out, dpi=300, bbox_inches='tight')
print(f"Wrote {out}")

# Print the data so you can cross-check against binary_table.md
print("\nMode / Median / Mean (km), P(wet):")
for i, (_, label) in enumerate(MODELS):
    for j, (_, vp) in enumerate(CASES):
        print(f"  {label:<25} {vp}  {mode_a[i,j]:5.2f} / {median_a[i,j]:5.2f} / {mean_a[i,j]:5.2f}   P_wet={pwet_a[i,j]:.2f}")
