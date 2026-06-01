"""Berryman self-consistent effective medium + Gassmann fluid substitution.

Direct port of `Inversion/berryscm.m` from
https://github.com/mattimorzfeld/WMM24
"""

from __future__ import annotations

import numpy as np


def berryscm(k, mu, asp, x, ro1, P_water):
    """Effective elastic moduli of an N-phase composite via Berryman SCM,
    followed by Gassmann fluid substitution.

    Parameters
    ----------
    k, mu : array_like, shape (N,)
        Bulk and shear moduli of each constituent phase (Pa).
    asp : array_like, shape (N,)
        Aspect ratio of inclusions for each phase. <1 oblate, >1 prolate.
        A value of exactly 1 is replaced by 0.99 (as in the MATLAB code).
    x : array_like, shape (N,)
        Volume fractions of each phase; must sum to 1.
    ro1 : float
        Dry bulk density (kg/m^3).
    P_water : float
        Fraction of pore space filled with water (0..1); remainder is gas.

    Returns
    -------
    kbr, mubr : float
        Berryman self-consistent dry bulk and shear moduli.
    vp, vs : float
        P- and S-wave velocities after fluid substitution (m/s).
    ro2 : float
        Bulk density after fluid substitution (kg/m^3).
    k2 : float
        Bulk modulus after Gassmann fluid substitution.
    """
    k = np.asarray(k, dtype=float).ravel()
    mu = np.asarray(mu, dtype=float).ravel()
    asp = np.asarray(asp, dtype=float).ravel().copy()
    x = np.asarray(x, dtype=float).ravel()

    # asp==1 is a removable singularity in the formulae below
    asp[asp == 1.0] = 0.99

    theta = np.zeros_like(asp)
    fn = np.zeros_like(asp)

    obl = asp < 1.0
    if np.any(obl):
        a = asp[obl]
        theta[obl] = (a / (1 - a**2) ** 1.5) * (np.arccos(a) - a * np.sqrt(1 - a**2))
        fn[obl] = (a**2 / (1 - a**2)) * (3 * theta[obl] - 2)

    pro = asp > 1.0
    if np.any(pro):
        a = asp[pro]
        theta[pro] = (a / (a**2 - 1) ** 1.5) * (a * np.sqrt(a**2 - 1) - np.arccosh(a))
        fn[pro] = (a**2 / (a**2 - 1)) * (2 - 3 * theta[pro])

    # Self-consistent iteration for effective dry moduli
    ksc = float(np.sum(k * x))
    musc = float(np.sum(mu * x))
    tol = 1e-6 * k[0]
    del_ = abs(ksc)  # ensure entry
    knew = 0.0
    munew = musc
    niter = 0
    while del_ > abs(tol) and niter < 3000:
        nusc = (3 * ksc - 2 * musc) / (2 * (3 * ksc + musc))
        a_ = mu / musc - 1.0
        b_ = (1.0 / 3.0) * (k / ksc - mu / musc)
        r = (1 - 2 * nusc) / (2 * (1 - nusc))

        f1 = 1 + a_ * (1.5 * (fn + theta) - r * (1.5 * fn + 2.5 * theta - 4.0 / 3.0))
        f2 = 1 + a_ * (1 + 1.5 * (fn + theta) - (r / 2) * (3 * fn + 5 * theta)) + b_ * (3 - 4 * r)
        f2 = f2 + (a_ / 2) * (a_ + 3 * b_) * (3 - 4 * r) * (
            fn + theta - r * (fn - theta + 2 * theta**2)
        )
        f3 = 1 + a_ * (1 - (fn + 1.5 * theta) + r * (fn + theta))
        f4 = 1 + (a_ / 4) * (fn + 3 * theta - r * (fn - theta))
        f5 = a_ * (-fn + r * (fn + theta - 4.0 / 3.0)) + b_ * theta * (3 - 4 * r)
        f6 = 1 + a_ * (1 + fn - r * (fn + theta)) + b_ * (1 - theta) * (3 - 4 * r)
        f7 = 2 + (a_ / 4) * (3 * fn + 9 * theta - r * (3 * fn + 5 * theta)) + b_ * theta * (3 - 4 * r)
        f8 = a_ * (1 - 2 * r + (fn / 2) * (r - 1) + (theta / 2) * (5 * r - 3)) + b_ * (1 - theta) * (3 - 4 * r)
        f9 = a_ * ((r - 1) * fn - r * theta) + b_ * theta * (3 - 4 * r)

        p = (3 * f1 / f2) / 3.0
        q = ((2.0 / f3) + (1.0 / f4) + (f4 * f5 + f6 * f7 - f8 * f9) / (f2 * f4)) / 5.0

        knew = float(np.sum(x * k * p) / np.sum(x * p))
        munew = float(np.sum(x * mu * q) / np.sum(x * q))

        del_ = abs(ksc - knew)
        ksc = knew
        musc = munew
        niter += 1

    kbr = ksc
    mubr = musc

    # Fluid mixture (gas + water) and Gassmann substitution
    rofl1 = 0.020      # gas density (kg/m^3)
    kfl1 = 0.0         # gas bulk modulus (Pa)
    ro_water = 1000.0
    k_water = 2.2e9
    P_gas = 1.0 - P_water
    rofl2 = P_gas * rofl1 + P_water * ro_water
    kfl2 = P_gas * kfl1 + P_water * k_water

    k0 = k[0]          # solid mineral bulk modulus
    phi = x[1]         # porosity
    k1 = kbr           # dry bulk modulus

    ro2 = ro1 - phi * rofl1 + phi * rofl2
    aa = k1 / (k0 - k1) - kfl1 / (phi * (k0 - kfl1)) + kfl2 / (phi * (k0 - kfl2))
    k2 = k0 * aa / (1 + aa)
    mu2 = mubr

    vp = np.sqrt((k2 + (4.0 / 3.0) * mu2) / ro2)
    vs = np.sqrt(mu2 / ro2)

    return kbr, mubr, vp, vs, ro2, k2
