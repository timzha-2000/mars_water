#!/usr/bin/env python3
"""Warm/hot-start re-run of a single rock-physics inversion, matching the
Wright et al. (2024) MCMC protocol, for the Appendix C initialization-robustness
check (the "warm start" paragraph).

For one executed notebook (which contains the EXACT forward model, bounds, data
vector, and emcee sampler that produced a published chain), this:

  1. exec()s the notebook's code cells up to (but not including) its
     `sampler.run_mcmc(...)` call, capturing the already-constructed emcee
     `sampler` object -- whose `log_prob_fn` has all args/globals baked in, so
     the procedure is identical across the two code frameworks in this study
     (log_post(theta,lb,ub,d,s,H) vs. log_probability(theta));
  2. runs Wright et al.'s protocol on that exact sampler:
       - 500-step cold start from the notebook's prior initialization,
       - keep only walkers with normalized misfit RMSE < 3, resample Ne of them,
       - 50,000-step production run from those warm starts,
       - discard the first 1,000 steps (BurnIn) as in their RunMe.m;
  3. saves the (phi, Sw, thickness) posterior so it can be compared against the
     cold-started chain in models/<model>/outputs_*/.

The executed notebook is produced by each model's run.py (e.g.
models/1_SCM_Voigt/bm_executed_B_wright_inherited.ipynb); analysis/launch_hotstart_all.sh
drives this over all 24 theory-velocity combinations.

Usage:  python3 run_hotstart.py <notebook.ipynb> <out.npz> [nsteps] [burnin] [cold] [rmse]
Faithfulness note: the reused sampler uses emcee's default stretch move (a=2),
the same as the published cold-start chains, so the only thing that differs
between this run and the published one is the initialization protocol.
"""
import json, sys, re, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import warnings; warnings.filterwarnings("ignore")
np.seterr(all="ignore")
import emcee

NB, OUT = sys.argv[1], sys.argv[2]
NSTEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 50000
BURNIN = int(sys.argv[4]) if len(sys.argv) > 4 else 1000
COLD   = int(sys.argv[5]) if len(sys.argv) > 5 else 500
RMSE_T = float(sys.argv[6]) if len(sys.argv) > 6 else 3.0
NDATA  = 3  # Vp, Vs, rho  -> RMSE = sqrt(chi2 / NDATA), matching RunMe.m


def load_namespace_upto_sampler(path):
    """exec the notebook's code cells until the sampler is built, stopping
    before any run_mcmc. Returns the resulting namespace."""
    nb = json.load(open(path))
    ns = {"__name__": "__hotstart__"}
    # make plotting inert in case a pre-sampler cell touches it
    exec("import matplotlib; matplotlib.use('Agg')\n"
         "import matplotlib.pyplot as _plt; _plt.show=lambda *a,**k:None", ns)
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if "run_mcmc" in src:
            # exec only the lines preceding the first run_mcmc call, then stop
            lines = src.split("\n")
            cut = next(i for i, l in enumerate(lines) if "run_mcmc" in l)
            head = "\n".join(lines[:cut])
            if head.strip():
                exec(head, ns)
            return ns
        exec(src, ns)
    return ns


def thickness_columns(path):
    """Read the (cx, cy) sample columns the notebook multiplies for thickness."""
    nb = json.load(open(path))
    src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    m = re.search(r"thickness_samples\s*=\s*8500\s*\*\s*samples\[:,\s*(\d+)\]\s*\*\s*samples\[:,\s*(\d+)\]", src)
    if not m:
        raise RuntimeError("could not find thickness column definition")
    return int(m.group(1)), int(m.group(2))


def find_init(ns, nwalkers, ndim):
    """Locate the notebook's initial walker array (nwalkers, ndim)."""
    for name in ("initial", "prior_pdf", "p0", "pos", "pos0", "start"):
        v = ns.get(name)
        if isinstance(v, np.ndarray) and v.shape == (nwalkers, ndim):
            return np.array(v, dtype=float)
    for v in ns.values():  # fallback: any array of the right shape
        if isinstance(v, np.ndarray) and v.shape == (nwalkers, ndim):
            return np.array(v, dtype=float)
    raise RuntimeError("no (nwalkers, ndim) initial array found in notebook")


def main():
    t0 = time.time()
    ns = load_namespace_upto_sampler(NB)
    sampler = ns["sampler"]
    ndim, nwalkers = sampler.ndim, sampler.nwalkers
    cx, cy = thickness_columns(NB)
    init = find_init(ns, nwalkers, ndim)
    rng = np.random.default_rng(20240611)

    # --- Wright protocol: cold start -> RMSE<3 -> resample ---
    sampler.run_mcmc(init, COLD, progress=False)
    Xc = sampler.get_chain(flat=True)
    Lc = sampler.get_log_prob(flat=True)
    rmse = np.sqrt(np.maximum(-2.0 * Lc / NDATA, 0.0))
    good = Xc[np.isfinite(Lc) & (rmse < RMSE_T)]
    if len(good) < nwalkers:               # fallback: take best-fitting walkers
        good = Xc[np.argsort(-Lc)[:max(nwalkers, 50)]]
    warm0 = good[rng.integers(0, len(good), nwalkers)]

    # --- production run from warm starts, then discard burn-in ---
    sampler.reset()
    sampler.run_mcmc(warm0, NSTEPS, progress=False)
    chain = sampler.get_chain(discard=BURNIN, flat=True)

    a, b = chain[:, cx], chain[:, cy]
    th = 8.5 * a * b                       # km  (8500 m * product / 1000)
    np.savez(OUT, col_a=a, col_b=b, th=th, cols=np.array([cx, cy]))
    print(f"{NB.split('/')[0]:10s} {NB.split('/')[-1][:24]:24s} "
          f"warm done {(time.time()-t0)/60:5.1f} min  N={len(th)}  "
          f"median={np.median(th):.3f} mean={np.mean(th):.3f} "
          f"P(<0.5)={np.mean(th<0.5):.3f}", flush=True)


if __name__ == "__main__":
    main()
