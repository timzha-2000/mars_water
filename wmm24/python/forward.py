"""Forward wrapper (myBerry) and log-posterior (myLogPi).

Port of `Inversion/myBerry.m` and `Inversion/myLogPi.m` from
https://github.com/mattimorzfeld/WMM24

Parameter vector theta = [asp, phi, P_water, k_GPa, mu_GPa, rho_min]:
    asp     : pore aspect ratio
    phi     : porosity (volume fraction)
    P_water : pore-space water saturation (0..1)
    k_GPa   : mineral bulk modulus (GPa)
    mu_GPa  : mineral shear modulus (GPa)
    rho_min : mineral density (kg/m^3)

Data vector convention (H selects which entries are used):
    d[0] = vp (km/s),  d[1] = vs (km/s),  d[2] = rho (kg/m^3)
"""

from __future__ import annotations

import numpy as np

from berryscm import berryscm


def my_berry(theta, H):
    """Forward model returning the data-vector entries selected by H."""
    asp = np.array([1.0, theta[0]])
    phi = theta[1]
    rock_vol = 1.0 - phi
    x = np.array([rock_vol, phi])
    rho_min = theta[5]
    rock_density = rho_min * rock_vol
    gas_density = 0.020 * phi
    rhob1 = rock_density + gas_density           # dry bulk density (kg/m^3)

    P_water = theta[2]
    k = np.array([theta[3] * 1e9, 0.0])          # GPa -> Pa
    mu = np.array([theta[4] * 1e9, 0.0])

    # Out-of-domain proposals (e.g., phi=0, asp>1 with imaginary acosh)
    # produce NaN here; my_log_pi catches those via the bounds check or
    # by returning -inf from the constraint tests, so just silence.
    with np.errstate(invalid="ignore", divide="ignore"):
        _, _, vp, vs, rhob, _ = berryscm(k, mu, asp, x, rhob1, P_water)

    H = np.asarray(H, dtype=int).ravel()
    sumH = int(H.sum())
    if sumH == 3:
        return np.array([vp / 1e3, vs / 1e3, rhob])
    if sumH == 2:
        if H[0] == 1 and H[1] == 1:
            return np.array([vp / 1e3, vs / 1e3])
        if H[0] == 1 and H[2] == 1:
            return np.array([vp / 1e3, rhob])
        if H[1] == 1 and H[2] == 1:
            return np.array([vs / 1e3, rhob])
        raise ValueError("Inconsistent H")
    if sumH == 1:
        if H[0] == 1:
            return np.array([vp / 1e3])
        if H[1] == 1:
            return np.array([vs / 1e3])
        if H[2] == 1:
            return np.array([rhob])
        raise ValueError("Inconsistent H")
    raise ValueError("Inconsistent H")


def my_log_pi(theta, lb, ub, d, s, H):
    """Return (log_posterior, model_prediction).

    Implements a uniform prior on the box [lb, ub] and a Gaussian
    likelihood with per-data-channel std `s`. Adds the physical
    constraints from the MATLAB code: vp > vs (when both are used),
    rho in (2500, 3100) (when rho appears alongside one of vp/vs alone
    or by itself).
    """
    theta = np.asarray(theta, dtype=float).ravel()
    lb = np.asarray(lb, dtype=float).ravel()
    ub = np.asarray(ub, dtype=float).ravel()
    d = np.asarray(d, dtype=float).ravel()
    s = np.asarray(s, dtype=float).ravel()
    H = np.asarray(H, dtype=int).ravel()
    sumH = int(H.sum())

    dM = my_berry(theta, H)

    # Box prior: strictly > lb, <= ub (matches MATLAB).
    if not (np.all(theta > lb) and np.all(theta <= ub)):
        return -np.inf, dM

    def gauss():
        r = (d - dM) / s
        return -0.5 * float(r @ r)

    if sumH == 2 and H[0] == 1 and H[1] == 1:
        # vp, vs -> require vp > vs
        return (gauss() if dM[0] > dM[1] else -np.inf), dM

    if sumH == 3:
        return (gauss() if dM[0] > dM[1] else -np.inf), dM

    if sumH == 2 and (H[0] == 1 or H[1] == 1):
        # one velocity + density: sanity-check the density
        # NB: dM[1] here is the density entry (vp,rho) or (vs,rho)
        rho = dM[1]
        return (gauss() if 2500.0 < rho < 3100.0 else -np.inf), dM

    if sumH == 1 and H[2] == 1:
        rho = dM[0]
        return (gauss() if 2500.0 < rho < 3100.0 else -np.inf), dM

    return gauss(), dM
