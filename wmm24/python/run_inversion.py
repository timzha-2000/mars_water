"""Driver script — port of `Inversion/RunMe.m` from the WMM24 repo.

Cold-start the ensemble inside the parameter bounds, run a short chain
to find a region of decent fit, resample walkers from that region, then
run a long warm-start chain. Print posterior mean +/- std for each
parameter at the end.

Usage:
    python run_inversion.py [warm_steps] [cold_steps]
"""

from __future__ import annotations

import sys

import numpy as np

from forward import my_log_pi
from hammer import my_hammer


def assign_data(d_full, s_full, H):
    """Select the subset of (d, s) indicated by H, matching AssignData.m."""
    H = np.asarray(H, dtype=int).ravel()
    sumH = int(H.sum())
    if sumH == 3:
        return d_full.copy(), s_full.copy()
    if sumH == 2:
        if H[0] == 1 and H[1] == 1:
            return d_full[[0, 1]], s_full[[0, 1]]
        if H[0] == 1 and H[2] == 1:
            return d_full[[0, 2]], s_full[[0, 2]]
        if H[1] == 1 and H[2] == 1:
            return d_full[[1, 2]], s_full[[1, 2]]
    if sumH == 1:
        if H[0] == 1:
            return d_full[[0]], s_full[[0]]
        if H[1] == 1:
            return d_full[[1]], s_full[[1]]
        if H[2] == 1:
            return d_full[[2]], s_full[[2]]
    raise ValueError("Invalid H")


def main(warm_steps=50_000, cold_steps=500, seed=0):
    rng = np.random.default_rng(seed)

    # Data: vp (km/s), vs (km/s), rho_m (kg/m^3) — matches RunMe.m
    d_full = np.array([4.1, 2.5, 2589.0])
    s_full = np.array([0.2, 0.3, 157.0])
    H = np.array([1, 1, 1])
    d, s = assign_data(d_full, s_full, H)
    nData = int(H.sum())

    # Parameter bounds: [asp, phi, water, k_GPa, mu_GPa, rho_min] — matches RunMe.m
    lb = np.array([0.001, 0.00, 0.0, 75.6, 25.6, 2680.0])
    ub = np.array([1.0,   0.50, 1.0, 80.0, 40.0, 2900.0])
    n = lb.size

    def logpi(x):
        return my_log_pi(x, lb, ub, d, s, H)

    # Initialize Ne = 3*n walkers uniformly inside bounds; reject any
    # initial draw that hits -inf (out-of-bounds or unphysical).
    Ne = 3 * n
    Xo = np.zeros((n, Ne))
    filled = 0
    while filled < Ne:
        xo = lb + (ub - lb) * rng.uniform(size=n)
        lp, _ = logpi(xo)
        if np.isfinite(lp):
            Xo[:, filled] = xo
            filled += 1

    a = 2.6

    # Cold start: find a region of decent fit
    print(f"Cold start: {cold_steps} steps, Ne={Ne}, n={n}")
    X, _, LogPi, _ = my_hammer(cold_steps, Xo, a, logpi, H, rng=rng)
    Xrs = X.reshape(n, -1)
    LogPirs = LogPi.reshape(-1)
    RMSE = np.sqrt(2 * LogPirs / d_full.size)
    mask = RMSE < 3.0
    if not np.any(mask):
        print("Warning: no cold-start samples with RMSE < 3; using all.")
        good = Xrs
    else:
        good = Xrs[:, mask]
    idx = rng.integers(0, good.shape[1], size=Ne)
    Xo = good[:, idx]

    # Warm start: the production run
    print(f"Warm start: {warm_steps} steps")
    X, D, LogPi, acc = my_hammer(warm_steps, Xo, a, logpi, H, rng=rng)
    print(f"Final acceptance ratio: {acc:.4f}")

    # Burn-in and flatten (MATLAB driver uses 1000 against 5e4 steps)
    BurnIn = min(1000, warm_steps // 4)
    Xrs = X[:, BurnIn:, :].reshape(n, -1)
    Drs = D[:, BurnIn:, :].reshape(nData, -1)
    LogPirs = LogPi[:, BurnIn:].reshape(-1)
    RMSE = np.sqrt(2 * LogPirs / d_full.size)

    mean = Xrs.mean(axis=1)
    std = Xrs.std(axis=1)
    names = ["Asp. ratio", "Porosity ", "Water    ",
             "K (GPa)  ", "mu (GPa) ", "rho_min  "]
    print()
    for name, m, sp in zip(names, mean, std):
        print(f"{name}: {m:8.4g} +/- {sp:.4g}")

    print(f"\nData-fit RMSE: mean={RMSE.mean():.3f}, "
          f"min={RMSE.min():.3f}, max={RMSE.max():.3f}")

    return dict(X=X, D=D, LogPi=LogPi, AccRatio=acc,
                Xrs=Xrs, Drs=Drs, RMSE=RMSE, lb=lb, ub=ub, d=d, s=s, H=H)


if __name__ == "__main__":
    warm = int(sys.argv[1]) if len(sys.argv) > 1 else 50_000
    cold = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    main(warm_steps=warm, cold_steps=cold)
