"""Two-panel ridgeline showing prior-set sensitivity for SCM+Voigt at Vp=4.1.

Left panel : binary thickness mixture posterior (each ridge weighted KDE +
             dry spike at 0 of height proportional to P(dry)).
Right panel: continuous thickness posterior (each ridge is the KDE of
             per-sample thickness from the freely-sampled S_w MCMC).

Three ridges per panel, one per prior set:
  1: Table 3 (this study; ~Wright 2024 narrow)
  2: Wider   (Wright 2024 sensitivity bounds)
  3: Reply   (Wright 2025 Reply to Xiao et al.)
"""
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

BASE = Path(__file__).resolve().parents[1]
APPDIR = BASE / "wmm24/matlab/outputs_appendix_priors"

# (label, color, continuous_npy, binary_wet_npy, logZ_wet, logZ_dry)
SETS = [
    ("1: Table 3 (this study)",
     "#0072B2",  # blue
     BASE / "models/1_SCM_Voigt/outputs_bm/thickness_samples_B_wright_inherited.npy",
     BASE / "models/1_SCM_Voigt/outputs_binary_wet/thickness_samples_B_wright_inherited.npy",
     BASE / "models/1_SCM_Voigt/outputs_binary_wet/logZ_B_wright_inherited.txt",
     BASE / "models/1_SCM_Voigt/outputs_binary_dry/logZ_B_wright_inherited.txt"),
    ("2: Wider (Wright 2024 sensitivity)",
     "#E69F00",  # orange
     APPDIR / "thickness_samples_wider_cont.npy",
     APPDIR / "thickness_samples_wider_binary_wet.npy",
     APPDIR / "logZ_wider_wet.txt",
     APPDIR / "logZ_wider_dry.txt"),
    ("3: Reply (Wright 2025)",
     "#009E73",  # green
     APPDIR / "thickness_samples_reply_cont.npy",
     APPDIR / "thickness_samples_reply_binary_wet.npy",
     APPDIR / "logZ_reply_wet.txt",
     APPDIR / "logZ_reply_dry.txt"),
]

XLIM = (0, 4.0)
X_EVAL = np.linspace(XLIM[0], XLIM[1], 800)
rng = np.random.default_rng(42)

def read_logZ(p):
    return float(open(p).read())

def p_wet(zw, zd):
    return 1.0 / (1.0 + math.exp(zd - zw))

n_sets = len(SETS)
row_height = 1.0
BIN_WIDTH = 0.02        # km, histogram bin width for the dry-mass bar

# Precompute densities and stats
binary_curves, cont_curves = {}, {}
for i, (label, color, cont_npy, bin_wet_npy, lzw, lzd) in enumerate(SETS):
    cont = np.load(cont_npy) / 1000.0
    sub = rng.choice(cont, size=min(50000, len(cont)), replace=False)
    cont_curves[i] = gaussian_kde(sub, bw_method=0.05)(X_EVAL)

    wet = np.load(bin_wet_npy) / 1000.0
    sub = rng.choice(wet, size=min(50000, len(wet)), replace=False)
    pw = p_wet(read_logZ(lzw), read_logZ(lzd))
    # UNCONDITIONAL mixture density:
    #   p(D) = P(wet) * p_wet(D)  +  P(dry) * delta(D)
    # Wet branch contributes a density of area P(wet); dry branch is a delta
    # at 0 which we render as a single histogram bar of width BIN_WIDTH and
    # height P(dry)/BIN_WIDTH (so the bar's area equals P(dry)).
    binary_curves[i] = dict(
        density=gaussian_kde(sub, bw_method=0.05)(X_EVAL),  # conditional p(D | S_w=1), area 1
        p_wet=pw, p_dry=1.0 - pw,
    )

# Per-panel max so the two panels share comparable per-row scaling.
# For the binary panel the row-max must include the dry bar's density
# (P(dry)/BIN_WIDTH), which typically dominates the wet KDE peak.
bin_row_max  = max(c["density"].max() for c in binary_curves.values())
cont_row_max = max(d.max() for d in cont_curves.values())

fig, (ax_b, ax_c) = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)

def draw(ax, top_density, top_kind):
    """top_density is dict: i -> dict(density, p_wet, p_dry)  if kind='binary'
                          : i -> ndarray (density)            if kind='cont'"""
    scale_max = bin_row_max if top_kind == "binary" else cont_row_max
    scale = row_height * 0.85 / max(scale_max, 1e-12)

    for i_top in range(n_sets - 1, -1, -1):  # ridge 1 at top
        baseline = (n_sets - 1 - i_top) * row_height  # top-down
        ax.axhline(baseline, color='grey', linewidth=0.4, zorder=0)
        # We actually want set 0 at top of the figure, so:
        # baseline for set i (0..n-1) at y = (n-1-i)*row_height
    for i in range(n_sets):
        baseline = (n_sets - 1 - i) * row_height
        label, color, *_ = SETS[i]
        if top_kind == "binary":
            c = top_density[i]
            y_vals = baseline + c["density"] * scale
            ax.fill_between(X_EVAL, baseline, y_vals, alpha=0.30, color=color)
            ax.plot(X_EVAL, y_vals, color=color, linewidth=1.8)
            ax.text(XLIM[1]*0.97, baseline + 0.05,
                    f"$P(S_w = 1) = {c['p_wet']:.2f}$",
                    ha='right', va='bottom', fontsize=11, color='black')
        else:
            density = top_density[i]
            y_vals = baseline + density * scale
            ax.fill_between(X_EVAL, baseline, y_vals, alpha=0.30, color=color)
            ax.plot(X_EVAL, y_vals, color=color, linewidth=1.8)

    ax.set_xlim(*XLIM)
    ax.set_ylim(-0.1, n_sets * row_height + 0.1)
    ax.tick_params(axis='x', which='major', labelsize=12, length=6, width=1.2)
    ax.tick_params(axis='y', which='major', length=0)
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)

draw(ax_b, binary_curves, "binary")
draw(ax_c, cont_curves,   "cont")

ax_b.set_title("Binary $S_w \\in \\{0, 1\\}$", fontsize=15)
ax_c.set_title("Continuous $S_w \\in [0, 1]$", fontsize=15)

# Shared y-tick labels (prior-set names) on the left panel
ax_b.set_yticks([(n_sets - 1 - i) * row_height for i in range(n_sets)])
ax_b.set_yticklabels([label for label, *_ in SETS], fontsize=12)
ax_b.set_xlabel("Water-layer thickness (km)", fontsize=13)
ax_c.set_xlabel("Water-layer thickness (km)", fontsize=13)

# Color legend on the binary panel
handles = [
    Line2D([0], [0], color=SETS[0][1], lw=2,  label=SETS[0][0]),
    Line2D([0], [0], color=SETS[1][1], lw=2,  label=SETS[1][0]),
    Line2D([0], [0], color=SETS[2][1], lw=2,  label=SETS[2][0]),
]
ax_b.legend(handles=handles, loc='upper right', fontsize=10, frameon=True,
            bbox_to_anchor=(0.97, 0.97))

fig.suptitle("Prior-box sensitivity of the water-layer thickness posterior  (SCM + Voigt,  $V_p = 4.1$ km/s)",
             fontsize=14, y=1.02)
plt.tight_layout()
out = BASE / "prior_sensitivity_ridgeline.png"
fig.savefig(out, dpi=300, bbox_inches="tight")
print(f"Wrote {out}")
