import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from pathlib import Path

BASE = Path("/home/ubuntu/Downloads/mars_clean")

MODELS = [
    ("bm_github",  "outputs_bm",    "Theory 1: Berryman + Voigt"),
    ("bmw_github", "outputs_bmw",   "Theory 2: Berryman + Hill"),
    ("HS",         "outputs_hsu",   "Theory 3: HS + Voigt"),
    ("HSh",        "outputs_hsu",   "Theory 4: HS + Hill"),
    ("DEMV",       "outputs_demni", "Theory 5: DEM + Voigt"),
    ("DEMVH",      "outputs_demni", "Theory 6: DEM + Hill"),
    ("kt",         "outputs_kt",    "Theory 7: Kuster\u2013Toks\u00f6z"),
    ("vrh",        "outputs_vrh",   "Theory 8: Voigt\u2013Reuss\u2013Hill"),
]

CASES = [
    ("A_constraints_away",  r"$V_p=3.8$ km/s", "#0072B2"),   # blue
    ("B_wright_inherited",  r"$V_p=4.1$ km/s", "#E69F00"),   # orange
    ("C_insight_marsquake", r"$V_p=4.7$ km/s", "#009E73"),   # green
]

XLIM = (0, 5.0)
X_EVAL = np.linspace(XLIM[0], XLIM[1], 500)

fig, axes = plt.subplots(4, 2, figsize=(12, 14), sharex=True)

for idx, (model_key, out_sub, title) in enumerate(MODELS):
    row, col = idx // 2, idx % 2
    ax = axes[row, col]

    for case_suffix, case_label, color in CASES:
        npy = BASE / model_key / out_sub / f"thickness_samples_{case_suffix}.npy"
        samples_km = np.load(npy) / 1000.0

        # Subsample for KDE speed (900k is too many)
        rng = np.random.default_rng(42)
        sub = rng.choice(samples_km, size=min(50000, len(samples_km)), replace=False)

        kde = gaussian_kde(sub, bw_method=0.05)
        density = kde(X_EVAL)
        ax.plot(X_EVAL, density, color=color, linewidth=1.8, label=case_label)
        ax.fill_between(X_EVAL, density, alpha=0.15, color=color)

    ax.set_xlim(*XLIM)
    ax.set_ylim(bottom=0)
    ax.set_title(title, fontsize=13, fontweight='bold', loc='left')
    ax.tick_params(axis='both', which='major', labelsize=12, length=6, width=1.2)

    if row == 3:
        ax.set_xlabel("water-layer thickness (km)", fontsize=14)
    if col == 0:
        ax.set_ylabel("probability density", fontsize=14)

# Single shared legend at top
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=14,
           frameon=True, bbox_to_anchor=(0.5, 0.995))

plt.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("thickness_overlay_2x4.png", dpi=300, bbox_inches='tight')
fig.savefig("thickness_overlay_2x4.pdf", bbox_inches='tight')
plt.close(fig)
print("Saved thickness_overlay_2x4.png and .pdf")
