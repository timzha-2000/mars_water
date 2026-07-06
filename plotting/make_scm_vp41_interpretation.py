#!/usr/bin/env python3
r"""SCM, Vp=4.1 (Wright config): two word-free panels (paper Fig. 3).

Left  : the Vp=4.1 water-layer-thickness posterior (h = 8.5*phi*Sw), with the
        area under the density split into EQUAL-WIDTH 0.5 km bands.
Right : the SAME bands shown on the (phi, Sw) crossplot. Each 0.5 km-wide
        thickness slice maps to a band whose AREA on the crossplot grows as the
        thickness drops, so the low-water / low-thickness band sprawls over a
        large area and looks sparse even though it carries the most probability
        -- the visual-density bias. Black curves are the band boundaries, i.e.
        evenly spaced iso-thickness contours h = 8.5*phi*Sw = 0.5, 1.0, ... km.

Uses the SCM+Voigt (Theory 1) Vp=4.1 posterior, i.e. the same data as Fig. 2:
models/1_SCM_Voigt/outputs_bm/{samples,thickness_samples}_B_wright_inherited.npy.
Requires models/1_SCM_Voigt/run.py to have been executed first.

Run from anywhere:  python plotting/make_scm_vp41_interpretation.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.stats import gaussian_kde
from pathlib import Path

# Repo root, resolved relative to this file (plotting/ is one level below root).
BASE = Path(__file__).resolve().parents[1] / "models" / "1_SCM_Voigt" / "outputs_bm"
SUF = "B_wright_inherited"
D = 8.5                      # mid-crust thickness (km)
STEP = 0.5                   # equal-WIDTH thickness band size (km)

s = np.load(BASE / f"samples_{SUF}.npy")
phi, sw = s[:, 1], s[:, 2]
th = np.load(BASE / f"thickness_samples_{SUF}.npy") / 1000.0   # km, exact Fig.2 array

# equal-WIDTH thickness band edges -> evenly spaced iso-thickness contours
# at 0.5, 1.0, 1.5, ... km (h = 8.5*phi*Sw = const)
edges = np.arange(0, th.max() + STEP, STEP)
NB = len(edges) - 1
inner = edges[1:-1]                                  # iso-thickness contour levels
cmap = ListedColormap(plt.cm.turbo(np.linspace(0.05, 0.95, NB)))
band_colors = [cmap(i) for i in range(NB)]

# ================= figure =================
fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 6.5))

# ---------------- left: thickness posterior, band-coloured (equal areas) ----------------
xg = np.linspace(0, edges[-1], 800)
dens = gaussian_kde(th, bw_method=0.05)(xg)          # all posterior samples (no subsample)
for i in range(NB):
    m = (xg >= edges[i]) & (xg <= edges[i + 1])
    axL.fill_between(xg[m], 0, dens[m], color=band_colors[i], alpha=0.85, lw=0)
axL.plot(xg, dens, color="k", lw=1.4)
axL.set_xlim(0, edges[-1]); axL.set_ylim(bottom=0)
axL.set_xlabel(r"Water-layer thickness  $h = 8.5\,\phi\,S_w$  (km)", fontsize=13)
axL.set_ylabel(r"Probability density  $p(h)$", fontsize=13)
axL.spines["top"].set_visible(False)
axL.spines["right"].set_visible(False)

# ---------------- right: equal-mass thickness bands on the crossplot ----------------
band = np.digitize(th, inner)                        # 0 .. NB-1
norm = BoundaryNorm(np.arange(NB + 1) - 0.5, NB)
# all posterior samples (no subsample); low alpha so the visual-density contrast
# is preserved without oversaturating the dense high-phi/high-Sw corner
sc = axR.scatter(phi, sw, c=band, cmap=cmap, norm=norm,
                 s=2, alpha=0.08, linewidths=0, rasterized=True)
phi_line = np.linspace(0.001, 0.5, 600)
for h in inner:
    swl = h / (D * phi_line)
    mm = swl <= 1.0
    axR.plot(phi_line[mm], swl[mm], color="k", lw=1.3, zorder=5)
cb = fig.colorbar(sc, ax=axR, fraction=0.046, pad=0.02,
                  ticks=np.arange(NB), boundaries=np.arange(NB + 1) - 0.5)
cb.ax.set_yticklabels([f"{edges[i]:.1f}–{edges[i+1]:.1f}" for i in range(NB)])
axR.set_xlim(0, 0.40); axR.set_ylim(0, 1.0)
axR.set_xlabel(r"Porosity $\phi$", fontsize=13)
axR.set_ylabel(r"Water saturation $S_w$", fontsize=13)

plt.tight_layout()
fig.savefig("scm_vp41_interpretation.png", dpi=200, bbox_inches="tight")
fig.savefig("scm_vp41_interpretation.pdf", bbox_inches="tight")
print("band edges (km):", np.round(edges, 3))
print("Saved scm_vp41_interpretation.png and .pdf")
