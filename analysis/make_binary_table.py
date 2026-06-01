#!/usr/bin/env python3
"""Build the R1.7 binary-saturation results table (3 cases x 8 theories).

For each theory x case it reports, from the two end-member runs:
  P(wet) = Z_wet / (Z_wet + Z_dry)         (prior-MC evidence; ~Wright Fig 1F)
  binary water-layer thickness mode / median / mean (km), where the
  posterior is the mixture: prob P(dry) -> 0 km, prob P(wet) -> 8500*phi
  (the wet-run porosity posterior). Burn-in: first 20% of each chain dropped.
"""
import os, math, numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASES = [("A_constraints_away", "A: Vp=3.8"),
         ("B_wright_inherited", "B: Vp=4.1"),
         ("C_insight_marsquake", "C: Vp=4.7")]
THEORIES = [
    ("1: SCM+Voigt", "models/1_SCM_Voigt"), ("2: SCM+Hill", "models/2_SCM_Hill"),
    ("3: HS+Voigt", "models/3_HS_Voigt"), ("4: HS+Hill", "models/4_HS_Hill"),
    ("5: DEM+Voigt", "models/5_DEM_Voigt"), ("6: DEM+Hill", "models/6_DEM_Hill"),
    ("7: Kuster-Toksoz", "models/7_KT"), ("8: VRH", "models/8_VRH"),
]
BURN = 0.20
NMIX = 200000


def read_logZ(path):
    try:
        v = float(open(path).read())
        return v
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


def stats(theory_dir, case):
    zw = read_logZ(os.path.join(ROOT, theory_dir, "outputs_binary_wet", f"logZ_{case}.txt"))
    zd = read_logZ(os.path.join(ROOT, theory_dir, "outputs_binary_dry", f"logZ_{case}.txt"))
    pw = p_wet(zw, zd)
    wpath = os.path.join(ROOT, theory_dir, "outputs_binary_wet", f"thickness_samples_{case}.npy")
    wet_km = np.load(wpath) / 1000.0
    wet_km = wet_km[int(BURN * len(wet_km)):]            # drop burn-in
    rng = np.random.default_rng(0)
    pw_eff = 0.0 if math.isnan(pw) else pw
    n_wet = int(round(pw_eff * NMIX))
    mix = np.concatenate([
        rng.choice(wet_km, size=n_wet, replace=True) if n_wet > 0 else np.array([]),
        np.zeros(NMIX - n_wet)])
    # mode via histogram
    counts, edges = np.histogram(mix, bins=120)
    mode = 0.5 * (edges[counts.argmax()] + edges[counts.argmax() + 1])
    return dict(pwet=pw, mode=mode, median=float(np.median(mix)),
                mean=float(mix.mean()), p5=float(np.percentile(mix, 5)),
                p95=float(np.percentile(mix, 95)),
                wet_median=float(np.median(wet_km)), zw=zw, zd=zd)


def main():
    rows = []
    md = ["# R1.7 Binary water-saturation results (two end-members: wet S_w=1 / dry S_w=0)\n",
          "Thickness in km. P(wet)=Z_wet/(Z_wet+Z_dry) from prior-MC evidence. "
          "Binary thickness posterior is the wet/dry mixture (dry contributes 0 km).\n"]
    for case, clabel in CASES:
        md.append(f"\n## Case {clabel}\n")
        md.append("| Theory | P(wet) | mode | median | mean | 5% | 95% |")
        md.append("|---|---|---|---|---|---|---|")
        for tlabel, td in THEORIES:
            try:
                st = stats(td, case)
            except FileNotFoundError:
                md.append(f"| {tlabel} | (pending) | | | | | |")
                continue
            md.append(f"| {tlabel} | {st['pwet']:.2f} | {st['mode']:.2f} | "
                      f"{st['median']:.2f} | {st['mean']:.2f} | {st['p5']:.2f} | {st['p95']:.2f} |")
            rows.append([case, tlabel, st['pwet'], st['mode'], st['median'],
                         st['mean'], st['p5'], st['p95'], st['wet_median'], st['zw'], st['zd']])
    out_md = os.path.join(ROOT, "analysis", "binary_table.md")
    open(out_md, "w").write("\n".join(md) + "\n")
    # CSV
    import csv
    out_csv = os.path.join(ROOT, "analysis", "binary_table.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case", "theory", "P_wet", "thick_mode_km", "thick_median_km",
                    "thick_mean_km", "thick_p5_km", "thick_p95_km", "wetonly_median_km",
                    "logZ_wet", "logZ_dry"])
        w.writerows(rows)
    print("\n".join(md))
    print(f"\nWrote {out_md} and {out_csv}")


if __name__ == "__main__":
    main()
