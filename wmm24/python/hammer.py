"""Affine-invariant ensemble (stretch-move) sampler.

Port of `myHammer.m`, `MoveEnsemble.m`, `sampleG.m`, `findPartner.m`
from https://github.com/mattimorzfeld/WMM24

The log-posterior callable is expected to return a (log_pi, data_pred)
tuple, matching `myLogPi`. Chains are stored as:

    X        : (ndim, nsteps, nwalkers)
    D        : (ndata, nsteps, nwalkers)
    LogPi    : (nwalkers, nsteps)        - stores -log_pi (i.e. chi^2/2),
                                           matching the MATLAB convention.

The per-sweep update is sequential (each walker proposed in turn using
a uniformly-chosen partner index from the *current* ensemble state),
matching the MATLAB implementation exactly.
"""

from __future__ import annotations

import sys

import numpy as np


def sample_g(n, a, rng):
    """Draw n samples from g(z) on [1/a, a] with g(z) ∝ 1/sqrt(z).

    Inverse-CDF form used in `sampleG.m`:
        z = ( 1/sqrt(a) + (a-1)/sqrt(a) * U )^2,  U ~ Uniform(0,1).
    """
    y = rng.uniform(size=n)
    C = 2.0 * (a - 1.0) / np.sqrt(a)
    return (1.0 / np.sqrt(a) + (C / 2.0) * y) ** 2


def find_partner(ind, Ne, rng):
    """Index of a random walker other than `ind`."""
    while True:
        pind = int(rng.integers(0, Ne))
        if pind != ind:
            return pind


def move_ensemble(cX, a, logpi, nData, rng):
    """One stretch-move sweep over the ensemble.

    cX     : (n, Ne) current ensemble (modified in place and returned).
    Returns (cX, D, clogpi, accMoves) where clogpi[j] = -logpi(cX[:,j]).
    """
    n, Ne = cX.shape
    z = sample_g(Ne, a, rng)
    accMoves = 0
    clogpi = np.zeros(Ne)
    D = np.zeros((nData, Ne))

    for jj in range(Ne):
        x = cX[:, jj].copy()
        pind = find_partner(jj, Ne, rng)
        xj = cX[:, pind]

        y = xj + z[jj] * (x - xj)              # stretch proposal
        logpiy, dy = logpi(y)
        logpix, dx = logpi(x)
        log_alpha = (n - 1) * np.log(z[jj]) + logpiy - logpix

        if np.log(rng.uniform()) < log_alpha:  # accept
            cX[:, jj] = y
            D[:, jj] = dy
            accMoves += 1
        else:
            D[:, jj] = dx

        lp_here, _ = logpi(cX[:, jj])
        clogpi[jj] = -lp_here

    return cX, D, clogpi, accMoves


def my_hammer(Nsteps, Xo, a, logpi, H, rng=None, progress=True):
    """Run the ensemble sampler.

    Parameters
    ----------
    Nsteps : int
        Number of sweeps.
    Xo : (n, Ne) array
        Initial ensemble; columns are walkers.
    a : float
        Stretch-move parameter (typical 2.0; the MATLAB driver uses 2.6).
    logpi : callable theta -> (log_pi, data_pred)
        Log-posterior; must return -inf outside support.
    H : array_like or int
        Either the data-indicator vector (sum == nData) or nData itself.

    Returns
    -------
    X        : (n, Nsteps, Ne)
    D        : (nData, Nsteps, Ne)
    LogPi    : (Ne, Nsteps)  - stores -log_pi at each step
    AccRatio : float
    """
    rng = np.random.default_rng() if rng is None else rng
    H = np.atleast_1d(np.asarray(H)).ravel()
    nData = int(H.sum()) if H.size > 1 else int(H[0])

    n, Ne = Xo.shape
    X = np.zeros((n, Nsteps, Ne))
    D = np.zeros((nData, Nsteps, Ne))
    LogPi = np.zeros((Ne, Nsteps))
    X[:, 0, :] = Xo

    accTotal = 0
    for step in range(Nsteps - 1):
        cX = X[:, step, :].copy()
        cX, DX, clogpi, accMoves = move_ensemble(cX, a, logpi, nData, rng)
        if np.any(np.isnan(cX)):
            print(f"NaN encountered at step {step}; stopping.", file=sys.stderr)
            break
        X[:, step + 1, :] = cX
        D[:, step + 1, :] = DX
        LogPi[:, step + 1] = clogpi
        accTotal += accMoves

        if progress and (step + 1) % 100 == 0:
            ratio = accTotal / ((step + 1) * Ne)
            print(f"  Acc. ratio at step {step + 1}/{Nsteps}: {ratio:.4f}",
                  end="\r", flush=True)

    if progress:
        print()
    return X, D, LogPi, accTotal / (Nsteps * Ne)
