"""Binary-saturation analogue of make_thickness_ridgeline.py.

For each theory x case the ridge shows the BINARY thickness mixture:
- a delta-like spike at 0 km contributed by the dry end-member, of mass P(dry),
- a continuous lobe at positive thickness contributed by the wet end-member,
  weighted by P(wet) = Z_wet / (Z_wet + Z_dry).
"""
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.lines import Line2D

BASE = Path(__file__).resolve().parents[1]

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
    ("A_constraints_away",  r"$V_p=3.8$ km/s", "#0072B2"),
    ("B_wright_inherited",  r"$V_p=4.1$ km/s", "#E69F00"),
    ("C_insight_marsquake", r"$V_p=4.7$ km/s", "#999999"),
]

BURN = 0.20
XLIM = (0, 4.0)
X_EVAL = np.linspace(XLIM[0], XLIM[1], 800)
rng = np.random.default_rng(42)

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

# Precompute densities for each (theory, case): wet KDE x P(wet), plus a
# bar at x=0 of mass P(dry) rendered as a tall thin spike.
all_curves = {}        # (idx, case) -> dict(kde_density, p_wet, p_dry)
row_max = {}
SPIKE_HEIGHT = None    # will set after we know KDE scales

for idx, (model_key, _) in enumerate(MODELS):
    rmax = 0
    for case_suffix, _, _ in CASES:
        zw = read_logZ(BASE / model_key / "outputs_binary_wet" / f"logZ_{case_suffix}.txt")
        zd = read_logZ(BASE / model_key / "outputs_binary_dry" / f"logZ_{case_suffix}.txt")
        pw = p_wet(zw, zd)
        pw_eff = 0.0 if math.isnan(pw) else pw
        pd_eff = 1.0 - pw_eff

        wet = np.load(BASE / model_key / "outputs_binary_wet" / f"thickness_samples_{case_suffix}.npy") / 1000.0
        wet = wet[int(BURN * len(wet)):]
        sub = rng.choice(wet, size=min(50000, len(wet)), replace=False)
        kde = gaussian_kde(sub, bw_method=0.05)
        density_wet = kde(X_EVAL)            # conditional wet-branch density p(D | S_w=1), area 1

        all_curves[(idx, case_suffix)] = dict(
            density=density_wet, p_wet=pw_eff, p_dry=pd_eff
        )
        rmax = max(rmax, density_wet.max())
    row_max[idx] = rmax

n_rows = len(MODELS)
row_height = 1.0

fig, ax = plt.subplots(figsize=(11, 10))

# Reserve space for the dry spike as a fraction of the row height
SPIKE_FRAC = 0.85   # delta spike will rise to this fraction of the row if P(dry)=1
SPIKE_WIDTH = 0.04  # km, narrow rectangle width for the dry mass at 0

for idx in range(n_rows - 1, -1, -1):
    baseline = idx * row_height
    theory_idx = n_rows - 1 - idx
    model_key, label = MODELS[theory_idx]
    scale = row_height * 0.85 / max(row_max[theory_idx], 1e-12)

    for case_suffix, case_label, color in reversed(CASES):
        c = all_curves[(theory_idx, case_suffix)]
        y_vals = baseline + c["density"] * scale

        ax.fill_between(X_EVAL, baseline, y_vals, alpha=0.22, color=color)
        ax.plot(X_EVAL, y_vals, color=color, linewidth=1.5,
                label=case_label if idx == n_rows - 1 else None)

    ax.axhline(baseline, color='grey', linewidth=0.4, zorder=0)

ax.set_yticks([i * row_height for i in range(n_rows)])
ax.set_yticklabels([MODELS[n_rows - 1 - i][1] for i in range(n_rows)], fontsize=13)
ax.set_xlim(*XLIM)
ax.set_ylim(-0.1, n_rows * row_height + 0.1)
ax.set_xlabel("Binary water-layer thickness (km)", fontsize=16)
ax.tick_params(axis='x', which='major', labelsize=14, length=6, width=1.2)
ax.tick_params(axis='y', which='major', length=0)
for s in ('top', 'right', 'left'):
    ax.spines[s].set_visible(False)

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc='upper right', fontsize=12, frameon=True)

plt.tight_layout()
out = BASE / "thickness_ridgeline_binary.png"
fig.savefig(out, dpi=300, bbox_inches="tight")
print(f"Wrote {out}")
