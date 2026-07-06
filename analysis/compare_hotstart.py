#!/usr/bin/env python3
"""Compare warm/hot-start (Wright protocol) posteriors against the cold-start
chains, for all 24 model x case combinations (Appendix C warm-start robustness
check -- reproduces the "at most 0.11 km, 23 of 24 <= 0.05 km" statement).

warm:  analysis/hotstart_outputs/<model>_<case>.npz        (from run_hotstart.py)
cold:  models/<model>/<outputs>/thickness_samples_<case>.npy   (from run.py, /1000 = km)

Prints per-combination median / mean / P(h<0.5) for cold vs warm and their
differences, plus the worst-case shift. Combinations whose warm run has not
finished yet (or whose cold chain has not been generated) are skipped and listed.

Run from anywhere:  python analysis/compare_hotstart.py
"""
from pathlib import Path

import numpy as np

# Repo root, resolved relative to this file (analysis/ is one level below root).
BASE = Path(__file__).resolve().parents[1]
HOT = BASE / "analysis" / "hotstart_outputs"

# (model directory under the repo, output subdirectory written by run.py)
MODELS = [
    ("models/1_SCM_Voigt", "outputs_bm"),
    ("models/2_SCM_Hill",  "outputs_bmw"),
    ("models/3_HS_Voigt",  "outputs_hsu"),
    ("models/4_HS_Hill",   "outputs_hsu"),
    ("models/5_DEM_Voigt", "outputs_demni"),
    ("models/6_DEM_Hill",  "outputs_demni"),
    ("models/7_KT",        "outputs_kt"),
    ("models/8_VRH",       "outputs_vrh"),
]
CASEF = {'A': 'A_constraints_away', 'B': 'B_wright_inherited', 'C': 'C_insight_marsquake'}

print(f"{'model/case':22s} {'cold med':>8s} {'warm med':>8s} {'dmed':>7s} "
      f"{'dmean':>7s} {'dP<0.5':>7s}")
worst_med = worst_mean = 0.0
pending = []
for model_dir, out_sub in MODELS:
    key = Path(model_dir).name             # unique per model, e.g. "1_SCM_Voigt"
    for cl, cf in CASEF.items():
        warm_f = HOT / f"{key}_{cl}.npz"
        cold_f = BASE / model_dir / out_sub / f"thickness_samples_{cf}.npy"
        try:
            tw = np.load(warm_f)['th']
            tc = np.load(cold_f) / 1000.0
        except FileNotFoundError:
            pending.append(f"{key}/{cl}")
            continue
        dmed = np.median(tw) - np.median(tc)
        dmean = np.mean(tw) - np.mean(tc)
        dpb = np.mean(tw < 0.5) - np.mean(tc < 0.5)
        worst_med = max(worst_med, abs(dmed)); worst_mean = max(worst_mean, abs(dmean))
        flag = '  <-- degenerate' if abs(dmed) > 0.05 else ''
        print(f"{key + '/' + cl:22s} {np.median(tc):8.3f} {np.median(tw):8.3f} "
              f"{dmed:+7.3f} {dmean:+7.3f} {dpb:+7.3f}{flag}")
print(f"\nLargest |dmedian| = {worst_med:.3f} km   |dmean| = {worst_mean:.3f} km")
if pending:
    print(f"Pending ({len(pending)}): {', '.join(pending)}")
    print("  (run each models/<m>/run.py, then analysis/launch_hotstart_all.sh)")
