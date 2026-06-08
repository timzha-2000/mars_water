#!/usr/bin/env python3
"""MCMC convergence diagnostics (Appendix C of the paper).

For every theory and velocity case this script loads the saved MCMC chain,
restores its (n_steps, n_walkers, n_dim) shape, and computes:
  - the integrated autocorrelation time tau (max over parameters),
  - the ratio n_steps / tau (rule-of-thumb threshold is 50),
  - the effective sample size ESS = n_walkers * n_steps / tau,
  - the split Gelman-Rubin statistic R-hat (max over parameters), computed by
    treating each of the n_walkers walkers as a chain and splitting it in half.

Together these reproduce the four entries of Supplementary Table S3. The R-hat
values were verified against the saved chains and match the table (e.g. the
single-non-mixing-walker cases DEM+Voigt/4.1 -> 2.28, DEM+Hill/4.7 -> 1.56,
KT/4.7 -> 1.12, and the well-converged HS combinations <= 1.007).

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


def split_rhat(chain):
    """Split Gelman-Rubin R-hat per parameter for a (n_steps, n_walkers, n_dim) chain.

    Each walker is treated as a chain and split in half, giving m = 2 * n_walkers
    half-chains of length n = n_steps // 2. For each parameter,
        B = n / (m - 1) * sum_j (mean_j - grand_mean)^2     (between-chain var)
        W = mean_j( var(half-chain j) )                     (within-chain var)
        var_hat = (n - 1) / n * W + B / n
        R-hat = sqrt(var_hat / W).
    Returns an array of R-hat values, one per parameter.
    """
    n_steps, n_walkers, n_dim = chain.shape
    n = n_steps // 2
    halves = np.concatenate([chain[:n], chain[n:2 * n]], axis=1)  # (n, 2*n_walkers, n_dim)
    m = halves.shape[1]
    chain_means = halves.mean(axis=0)               # (m, n_dim)
    grand_mean = chain_means.mean(axis=0)           # (n_dim,)
    B = n / (m - 1) * np.sum((chain_means - grand_mean) ** 2, axis=0)
    W = halves.var(axis=0, ddof=1).mean(axis=0)     # (n_dim,)
    var_hat = (n - 1) / n * W + B / n
    return np.sqrt(var_hat / W)


def diagnostics(flat_chain):
    """Return (tau_max, n_steps, n_walkers, rhat_max, reliable) from a flat chain."""
    n_dim = flat_chain.shape[1]
    n_walkers = 3 * n_dim
    n_steps = flat_chain.shape[0] // n_walkers
    chain = flat_chain.reshape(n_steps, n_walkers, n_dim)  # (steps, walkers, dim)
    tau = emcee.autocorr.integrated_time(chain, quiet=True)
    tau_max = float(np.max(tau))
    rhat_max = float(np.max(split_rhat(chain)))
    return tau_max, n_steps, n_walkers, rhat_max, (n_steps / tau_max) >= THRESHOLD


def main():
    rows = []
    print(f"{'Theory':18} {'Case':5} {'tau_max':>8} {'N/tau':>7} {'ESS':>7} {'Rhat':>6}  flag")
    print("-" * 64)
    for label, model_dir, out_sub in MODELS:
        for vp, suffix in CASES:
            f = BASE / model_dir / out_sub / f"samples_{suffix}.npy"
            if not f.exists():
                print(f"{label:18} {vp:5} {'--- missing: run models/' + model_dir.split('/')[-1] + '/run.py ---'}")
                continue
            tau, n_steps, n_walkers, rhat, reliable = diagnostics(np.load(f))
            ratio = round(n_steps / tau)
            ess = round(n_walkers * n_steps / tau)
            flag = "" if reliable else "*  (N/tau < 50; tau is a lower bound)"
            rows.append((label, vp, round(tau), ratio, ess, rhat, reliable))
            print(f"{label:18} {vp:5} {round(tau):8d} {ratio:7d} {ess:7d} {rhat:6.2f}  {flag}")

    # Optional CSV next to this script
    out_csv = BASE / "analysis" / "convergence_diagnostics.csv"
    with open(out_csv, "w") as fh:
        fh.write("theory,Vp_kms,tau_max,N_steps_over_tau,ESS,Rhat,reliable\n")
        for label, vp, tau, ratio, ess, rhat, reliable in rows:
            fh.write(f"{label},{vp},{tau},{ratio},{ess},{rhat:.3f},{int(reliable)}\n")
    print(f"\nWrote {out_csv.relative_to(BASE)}")


if __name__ == "__main__":
    main()
