#!/usr/bin/env python3
"""MCMC convergence diagnostics (Appendix C of the paper).

For every theory and velocity case this script loads the saved MCMC chain,
restores its (n_steps, n_walkers, n_dim) shape, and computes:
  - the integrated autocorrelation time tau (max over parameters),
  - the ratio n_steps / tau (rule-of-thumb threshold is 50),
  - the effective sample size ESS = n_walkers * n_steps / tau.

The samples were saved flattened with emcee's get_chain(flat=True), i.e. with
shape (n_walkers * n_steps, n_dim) in step-major order, so reshaping to
(n_steps, n_walkers, n_dim) exactly recovers the ensemble for emcee's
autocorrelation estimator. Each model is run with n_walkers = 3 * n_dim.

Run from anywhere:  python analysis/compute_convergence.py
Requires each model's `run.py` to have been executed first so that the
`samples_<case>.npy` files exist under models/<model>/<outputs>/.
"""
from pathlib import Path

import numpy as np
import emcee

# Repo root, resolved relative to this file (analysis/ is one level below root).
BASE = Path(__file__).resolve().parents[1]

# (label, model directory under the repo, output subdirectory written by run.py)
MODELS = [
    ("1: SCM + Voigt", "models/1_SCM_Voigt", "outputs_bm"),
    ("2: SCM + Hill",  "models/2_SCM_Hill",  "outputs_bmw"),
    ("3: HS + Voigt",  "models/3_HS_Voigt",  "outputs_hsu"),
    ("4: HS + Hill",   "models/4_HS_Hill",   "outputs_hsu"),
    ("5: DEM + Voigt", "models/5_DEM_Voigt", "outputs_demni"),
    ("6: DEM + Hill",  "models/6_DEM_Hill",  "outputs_demni"),
    ("7: Kuster-Toksoz", "models/7_KT",      "outputs_kt"),
    ("8: VRH",         "models/8_VRH",       "outputs_vrh"),
]

CASES = [
    ("3.8", "A_constraints_away"),
    ("4.1", "B_wright_inherited"),
    ("4.7", "C_insight_marsquake"),
]

THRESHOLD = 50  # n_steps / tau threshold above which tau is considered reliable


def diagnostics(flat_chain):
    """Return (tau_max, n_steps, n_walkers, reliable) from a flattened chain."""
    n_dim = flat_chain.shape[1]
    n_walkers = 3 * n_dim
    n_steps = flat_chain.shape[0] // n_walkers
    chain = flat_chain.reshape(n_steps, n_walkers, n_dim)  # (steps, walkers, dim)
    tau = emcee.autocorr.integrated_time(chain, quiet=True)
    tau_max = float(np.max(tau))
    return tau_max, n_steps, n_walkers, (n_steps / tau_max) >= THRESHOLD


def main():
    rows = []
    print(f"{'Theory':18} {'Case':5} {'tau_max':>8} {'N/tau':>7} {'ESS':>7}  flag")
    print("-" * 56)
    for label, model_dir, out_sub in MODELS:
        for vp, suffix in CASES:
            f = BASE / model_dir / out_sub / f"samples_{suffix}.npy"
            if not f.exists():
                print(f"{label:18} {vp:5} {'--- missing: run models/' + model_dir.split('/')[-1] + '/run.py ---'}")
                continue
            tau, n_steps, n_walkers, reliable = diagnostics(np.load(f))
            ratio = round(n_steps / tau)
            ess = round(n_walkers * n_steps / tau)
            flag = "" if reliable else "*  (N/tau < 50; tau is a lower bound)"
            rows.append((label, vp, round(tau), ratio, ess, reliable))
            print(f"{label:18} {vp:5} {round(tau):8d} {ratio:7d} {ess:7d}  {flag}")

    # Optional CSV next to this script
    out_csv = BASE / "analysis" / "convergence_diagnostics.csv"
    with open(out_csv, "w") as fh:
        fh.write("theory,Vp_kms,tau_max,N_steps_over_tau,ESS,reliable\n")
        for label, vp, tau, ratio, ess, reliable in rows:
            fh.write(f"{label},{vp},{tau},{ratio},{ess},{int(reliable)}\n")
    print(f"\nWrote {out_csv.relative_to(BASE)}")


if __name__ == "__main__":
    main()
