"""Run the Berryman WMM24 inversion for cases A/B/C and save:
    samples_<case>.npy            shape (N_walkers * N_steps, 6)
    thickness_samples_<case>.npy  shape (N_walkers * N_steps,)

The output schema matches the existing notebook workflow in
`bm_github/` so that the ridgeline / montage / barplot scripts read
these files without modification.

Thickness transform (from `Berryman_mod.ipynb`):
    thickness_m = 8500 * water_saturation * porosity
                 = 8500 * samples[:, 2] * samples[:, 1]
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

from forward import my_log_pi
from hammer import my_hammer


# (Vp_kms, Vs_kms, rho), (sigVp_kms, sigVs_kms, sigRho)
CASES = [
    {
        "name": "A_constraints_away",
        "d": (3.8, 2.2, 2589.0),
        "s": (1.0, 0.4, 157.0),
    },
    {
        "name": "B_wright_inherited",
        "d": (4.1, 2.5, 2589.0),
        "s": (0.2, 0.3, 157.0),
    },
    {
        "name": "C_insight_marsquake",
        "d": (4.7, 2.7, 2589.0),
        "s": (0.3, 0.1, 157.0),
    },
]

# Parameter bounds: [asp, phi, water, k_GPa, mu_GPa, rho_min]
LB = np.array([0.001, 0.0, 0.0, 75.6, 25.6, 2680.0])
UB = np.array([1.0,   0.5, 1.0, 80.0, 40.0, 2900.0])

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs_bm_py")
COLUMN_STRETCH_THICKNESS_M = 8500.0  # crustal saturated-layer assumption


def run_case(case, nsteps_warm, nsteps_cold, seed, out_dir):
    name = case["name"]
    d = np.asarray(case["d"], dtype=float)
    s = np.asarray(case["s"], dtype=float)
    H = np.array([1, 1, 1], dtype=int)

    rng = np.random.default_rng(seed)
    n = LB.size
    Ne = 3 * n
    a = 2.6

    def logpi(x):
        return my_log_pi(x, LB, UB, d, s, H)

    # Initialize Ne walkers uniformly inside the box, rejecting -inf
    Xo = np.zeros((n, Ne))
    filled = 0
    while filled < Ne:
        xo = LB + (UB - LB) * rng.uniform(size=n)
        lp, _ = logpi(xo)
        if np.isfinite(lp):
            Xo[:, filled] = xo
            filled += 1

    # Cold start: short chain, then resample walkers with RMSE < 3
    print(f"[{name}] cold start: {nsteps_cold} steps")
    X, _, LogPi, _ = my_hammer(nsteps_cold, Xo, a, logpi, H, rng=rng, progress=False)
    Xrs = X.reshape(n, -1)
    LogPirs = LogPi.reshape(-1)
    RMSE = np.sqrt(2 * LogPirs / d.size)
    mask = RMSE < 3.0
    good = Xrs[:, mask] if np.any(mask) else Xrs
    idx = rng.integers(0, good.shape[1], size=Ne)
    Xo = good[:, idx]

    # Warm start: production chain
    print(f"[{name}] warm start: {nsteps_warm} steps")
    t0 = time.time()
    X, _, _, acc = my_hammer(nsteps_warm, Xo, a, logpi, H, rng=rng, progress=False)
    print(f"[{name}] done in {time.time() - t0:.1f}s, acceptance={acc:.3f}")

    # Flatten to (N_walkers * N_steps, 6) — matches emcee get_chain(flat=True)
    # X is (n, Nsteps, Ne). emcee's order is (Nsteps, Ne, n) -> flat is
    # (Nsteps*Ne, n). Match that order: transpose to (Nsteps, Ne, n) first.
    flat = np.transpose(X, (1, 2, 0)).reshape(-1, n)

    # Thickness (m): 8500 * water_sat * porosity
    thickness = COLUMN_STRETCH_THICKNESS_M * flat[:, 2] * flat[:, 1]

    os.makedirs(out_dir, exist_ok=True)
    samples_path = os.path.join(out_dir, f"samples_{name}.npy")
    thickness_path = os.path.join(out_dir, f"thickness_samples_{name}.npy")
    np.save(samples_path, flat)
    np.save(thickness_path, thickness)
    print(f"[{name}] saved {samples_path} {flat.shape}")
    print(f"[{name}] saved {thickness_path} {thickness.shape}  "
          f"mean={thickness.mean():.1f} m  "
          f"5/95% = {np.percentile(thickness, [5, 95])}")
    return flat, thickness


def main():
    warm = int(sys.argv[1]) if len(sys.argv) > 1 else 50_000
    cold = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    case_filter = sys.argv[3] if len(sys.argv) > 3 else None

    cases = CASES if case_filter is None else [c for c in CASES if c["name"] == case_filter]
    if not cases:
        print(f"Unknown case '{case_filter}'", file=sys.stderr)
        sys.exit(1)

    for i, case in enumerate(cases):
        # Distinct seed per case so chains are independent but reproducible.
        run_case(case, warm, cold, seed=1000 + i, out_dir=OUT_DIR)


if __name__ == "__main__":
    main()
