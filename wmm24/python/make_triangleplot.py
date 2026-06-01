"""Corner / triangle plot for the WMM24 Python-port posterior samples.

Reproduces the figure produced by `Inversion/TrianglePlot.m` +
`PlotScript.m`: a 6x6 corner showing per-parameter marginal histograms
on the diagonal and smoothed 2D histograms below the diagonal.

Reads:  outputs_bm_py/samples_<case>.npy   (shape: N x 6)
Writes: outputs_bm_py/triangleplot_<case>.{pdf,png}
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import corner

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs_bm_py")

CASES = [
    ("A_constraints_away",  r"Case A:  $V_p=3.8$ km/s"),
    ("B_wright_inherited",  r"Case B:  $V_p=4.1$ km/s  (Wright et al. inherited)"),
    ("C_insight_marsquake", r"Case C:  $V_p=4.7$ km/s"),
]

LABELS = [r"$\alpha$",
          r"$\phi$",
          r"$\gamma_w$",
          r"$\kappa_m$ [GPa]",
          r"$\mu_m$ [GPa]",
          r"$\rho_m$ [kg/m$^3$]"]

# Bounds used in the inversion — used as plot ranges so cases are
# directly comparable.
RANGE = [(0.001, 1.0),
         (0.0, 0.5),
         (0.0, 1.0),
         (75.6, 80.0),
         (25.6, 40.0),
         (2680.0, 2900.0)]


def plot_case(case_suffix: str, title: str) -> str:
    samples = np.load(os.path.join(OUT_DIR, f"samples_{case_suffix}.npy"))
    # Drop a short burn-in just in case (the warm-start chain has already
    # had the cold-start dropped, but a few thousand more is cheap).
    if samples.shape[0] > 200_000:
        samples = samples[50_000:]

    fig = corner.corner(
        samples,
        labels=LABELS,
        range=RANGE,
        bins=40,
        smooth=1.0,
        plot_datapoints=False,
        plot_density=True,
        fill_contours=True,
        levels=(0.39, 0.86),                # 1- and 2-sigma in 2D
        color="#1f3a93",
        hist_kwargs=dict(color="#1f3a93", lw=1.5),
        show_titles=True,
        title_kwargs=dict(fontsize=10),
        label_kwargs=dict(fontsize=13),
        title_fmt=".3g",
    )
    fig.suptitle(title, fontsize=14)
    fig.subplots_adjust(top=0.95)

    out_pdf = os.path.join(OUT_DIR, f"triangleplot_{case_suffix}.pdf")
    out_png = os.path.join(OUT_DIR, f"triangleplot_{case_suffix}.png")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_pdf


def main() -> None:
    for case, title in CASES:
        out = plot_case(case, title)
        print(f"saved {out}")


if __name__ == "__main__":
    main()
