#!/usr/bin/env python3
"""R1.10 spread analysis: why the intermediate-Vp dataset admits the most water.

Reviewer R1.10 asks for a qualitative explanation of why intermediate Vp (rather
than higher or lower) yields the greatest amount of inferred water. Following
Per's reframing, we do NOT try to explain this mechanistically. Instead we show
quantitatively that the intermediate-Vp dataset (Vp = 4.1 km/s, case B) has the
LOWEST Vp/Vs ratio -- and that Vp/Vs, the standard fluid indicator, places those
observations in a region of parameter space where the rock-physics models admit a
WIDER range of phi*Sw combinations. The take-home is not a physical mechanism but
the sensitivity itself: inferred water content depends strongly on which seismic
dataset is used.

We quantify "wider range of phi*Sw combinations" two complementary ways, both
read off the per-sample water-layer thickness (thickness = D_midcrust * phi * Sw,
with D_midcrust = 8.5 km a fixed scalar, so phi*Sw = thickness_km / D_MIDCRUST):

  (1) ACROSS model families -- inter-model disagreement: range and coefficient of
      variation (CoV) of the per-theory median phi*Sw. This backs "which model
      family you pick matters."

  (2) WITHIN each model family -- intra-model posterior width: the per-theory
      central 90% width (p95-p5) of phi*Sw, summarized as the median over the 8
      theories. This backs "each family admits a wider range of combinations."

Both are linearly tied to thickness, so the conclusion is identical whether
stated in phi*Sw or in km of water.
"""
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
D_MIDCRUST_KM = 8.5  # fixed mid-crust thickness; thickness_km = D_MIDCRUST_KM * phi * Sw

# 8 theories (the standalone MATLAB-port run is intentionally excluded).
MODELS = [
    ("models/1_SCM_Voigt",  "outputs_bm"),
    ("models/2_SCM_Hill", "outputs_bmw"),
    ("models/3_HS_Voigt",         "outputs_hsu"),
    ("models/4_HS_Hill",        "outputs_hsu"),
    ("models/5_DEM_Voigt",       "outputs_demni"),
    ("models/6_DEM_Hill",      "outputs_demni"),
    ("models/7_KT",         "outputs_kt"),
    ("models/8_VRH",        "outputs_vrh"),
]

# (case suffix, Vp, Vs, sigma_Vp, sigma_Vs)  [km/s]; values from each run.py.
CASES = [
    ("A_constraints_away",  3.8, 2.2, 1.0, 0.4),
    ("B_wright_inherited",  4.1, 2.5, 0.2, 0.3),
    ("C_insight_marsquake", 4.7, 2.7, 0.3, 0.1),
]


def phi_sw_samples(case_suffix):
    """Return {model_label: phi*Sw sample array} for one case."""
    out = {}
    for d, sub in MODELS:
        th_km = np.load(BASE / d / sub / f"thickness_samples_{case_suffix}.npy") / 1000.0
        out[d] = th_km / D_MIDCRUST_KM  # phi*Sw, dimensionless
    return out


def main():
    print("Observed datasets (Vp/Vs is the fluid indicator; lowest -> most water):")
    print(f"  {'case':22} {'Vp':>4} {'Vs':>4} {'Vp/Vs':>6} {'sigVp':>6} {'sigVs':>6}")
    for nm, vp, vs, svp, svs in CASES:
        print(f"  {nm:22} {vp:4.1f} {vs:4.1f} {vp/vs:6.3f} {svp:6.2f} {svs:6.2f}")

    print("\n(1) ACROSS model families -- spread of per-theory MEDIAN phi*Sw:")
    print(f"  {'case':22} {'Vp/Vs':>6} {'min':>7} {'max':>7} {'range':>7} {'CoV%':>6}")
    for nm, vp, vs, svp, svs in CASES:
        meds = np.array([np.median(s) for s in phi_sw_samples(nm).values()])
        cov = 100 * meds.std() / meds.mean()
        print(f"  {nm:22} {vp/vs:6.3f} {meds.min():7.3f} {meds.max():7.3f} "
              f"{meds.max() - meds.min():7.3f} {cov:6.0f}")

    print("\n(2) WITHIN model families -- median over theories of the per-theory")
    print("    central 90% (p95-p5) posterior width of phi*Sw:")
    print(f"  {'case':22} {'Vp/Vs':>6} {'med 90% width':>14}")
    for nm, vp, vs, svp, svs in CASES:
        widths = np.array([np.percentile(s, 95) - np.percentile(s, 5)
                           for s in phi_sw_samples(nm).values()])
        print(f"  {nm:22} {vp/vs:6.3f} {np.median(widths):14.3f}")

    print("\nFor reference, the same spreads as water-layer thickness (km) "
          f"= {D_MIDCRUST_KM} km * phi*Sw:")
    print(f"  {'case':22} {'med-range km':>13} {'med 90% width km':>17}")
    for nm, vp, vs, svp, svs in CASES:
        s = phi_sw_samples(nm)
        meds = np.array([np.median(v) for v in s.values()]) * D_MIDCRUST_KM
        widths = np.array([(np.percentile(v, 95) - np.percentile(v, 5)) * D_MIDCRUST_KM
                           for v in s.values()])
        print(f"  {nm:22} {meds.max() - meds.min():13.2f} {np.median(widths):17.2f}")


if __name__ == "__main__":
    main()
