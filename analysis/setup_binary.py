#!/usr/bin/env python3
"""R1.7 binary water-saturation setup -- TWO END-MEMBER design.

Faithful to Wright, Morzfeld & Manga (2025), "Reply to Xiao et al.":
S_w is NOT sampled. Instead it is FIXED to an end-member (wet S_w=1 or
dry S_w=0) inside each theory's forward model, and the ensemble MCMC
samples the remaining parameters. We run every theory x case twice
(wet, dry) and compare the two via prior-Monte-Carlo evidence Z, so the
wet/dry probability split P(wet)=Z_wet/(Z_wet+Z_dry) reproduces their
Fig. 1F. Each theory's own run.py does the (case) data patching; we only
swap the notebook + output dir, fix S_w, and append a results tail.

Generates, per theory and end-member em in {wet,dry}:
  <dir>/<stem>_binary_<em>.ipynb
  <dir>/run_binary_<em>.py   (writes to <dir>/outputs_binary_<em>/)

Usage: python setup_binary.py [NSTEPS] [EVN]
  NSTEPS overrides chain length (smoke tests). EVN overrides prior-MC size.
"""
import os, re, sys, nbformat

# --------------------------------------------------------------------------
# Configuration / CLI
# --------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # repo root; each theory is a subdir
# Optional command-line overrides (both default to None = "use the real values"):
#   argv[1] NSTEPS -> shrink the MCMC chain length for a fast smoke test
#   argv[2] EVN    -> shrink the prior-Monte-Carlo evidence sample size
NSTEPS = int(sys.argv[1]) if len(sys.argv) > 1 else None
EVN_CLI = int(sys.argv[2]) if len(sys.argv) > 2 else None

# --------------------------------------------------------------------------
# The 8 rock-physics theories. Each dict is a recipe telling make()/tail()
# how to patch that theory's notebook. Fields:
#   key   : human-readable label (theory number + model)
#   d     : subdirectory under ROOT containing the theory's notebook + run.py
#   nb    : the source notebook to clone
#   run   : the theory's existing runner script to clone
#   style : "ds" -> notebook exposes log_post(theta, lb, ub, d, s, H);
#                   prior bounds come from the notebook's own lb/ub arrays
#           "obs"-> notebook exposes log_probability(x); no lb/ub in scope,
#                   so we must supply explicit prior bounds (pb_lb/pb_ub below)
#   swcol : index of the S_w parameter in the theta/x vector (the column we
#           pin to an interior value during prior sampling so bound checks pass)
#   evn   : default prior-MC evidence sample size for this theory
#           (DEM uses only 20k because its forward model is expensive)
#   repl  : (old_code, new_template) -- the exact source line to find in the
#           notebook and the replacement that FIXES S_w to the end-member.
#           "{SW}" gets filled with 1.0 (wet) or 0.0 (dry).
#   pb_lb/pb_ub : ("obs" style only) prior box for the evidence integral.
# --------------------------------------------------------------------------
THEORIES = [
    dict(key="1_SCM_Voigt", d="models/1_SCM_Voigt",  nb="Berryman_mod.ipynb",      run="run.py",
         style="ds", swcol=2, evn=100000,
         repl=("P_water = theta[2]", "P_water = {SW}  # R1.7 fixed end-member S_w")),
    dict(key="2_SCM_Hill",  d="models/2_SCM_Hill", nb="Berryman_mod_wood.ipynb", run="run.py",
         style="ds", swcol=2, evn=100000,
         repl=("P_water = theta[2]", "P_water = {SW}  # R1.7 fixed end-member S_w")),
    dict(key="3_HS_Voigt",  d="models/3_HS_Voigt",  nb="HS.ipynb",  run="run.py",
         style="ds", swcol=2, evn=100000,
         repl=("_, phi, Sw, Kmin, Gmin, rho_m = theta",
               "_, phi, Sw, Kmin, Gmin, rho_m = theta\n    Sw = {SW}  # R1.7 fixed end-member S_w")),
    dict(key="4_HS_Hill",   d="models/4_HS_Hill", nb="HSh.ipynb", run="run.py",
         style="ds", swcol=2, evn=100000,
         repl=("_, phi, Sw, Kmin, Gmin, rho_m = theta",
               "_, phi, Sw, Kmin, Gmin, rho_m = theta\n    Sw = {SW}  # R1.7 fixed end-member S_w")),
    dict(key="5_DEM_Voigt", d="models/5_DEM_Voigt",  nb="DEM.ipynb",  run="run.py",
         style="ds", swcol=2, evn=20000,
         repl=("P_water = theta[2]", "P_water = {SW}  # R1.7 fixed end-member S_w")),
    dict(key="6_DEM_Hill",  d="models/6_DEM_Hill", nb="DEMH.ipynb", run="run.py",
         style="ds", swcol=2, evn=20000,
         repl=("P_water = theta[2]", "P_water = {SW}  # R1.7 fixed end-member S_w")),
    dict(key="7_KT", d="models/7_KT", nb="kt.ipynb", run="run.py",
         style="obs", swcol=0, evn=100000,
         pb_lb="[0.0, 0.001, -3.0, 756e8, 256e8, 2680.0]",
         pb_ub="[1.0, 0.5,    0.0, 80e9,  40e9,  2900.0]",
         repl=("S_w, phi, logalpha, K_m, mu_m, rho_m = x\n    alpha = 10**(logalpha)",
               "S_w, phi, logalpha, K_m, mu_m, rho_m = x\n    S_w = {SW}  # R1.7 fixed end-member S_w\n    alpha = 10**(logalpha)")),
    dict(key="8_VRH", d="models/8_VRH", nb="vrh.ipynb", run="run.py",
         style="obs", swcol=0, evn=100000,
         pb_lb="[0.0, 0.001, 756e8, 256e8, 2680.0]",
         pb_ub="[1.0, 0.5,   80e9,  40e9,  2900.0]",
         repl=("S_w, phi, K_s, G_s, rho_s = x\n    K_f = S_w * K_water + (1 - S_w) * K_gas",
               "S_w, phi, K_s, G_s, rho_s = x\n    S_w = {SW}  # R1.7 fixed end-member S_w\n    K_f = S_w * K_water + (1 - S_w) * K_gas")),
]

# The two end-members. Key = label used in filenames/dirs; value = the literal
# S_w value spliced into the forward model. We run BOTH for every theory x case.
EM = {"wet": "1.0", "dry": "0.0"}


def tail(th, sw_literal, em, evn):
    """Return Python source (as a string) for an extra notebook cell that runs
    AFTER the MCMC has finished. It does two jobs:
      (1) save the chain + a binary water-layer THICKNESS sample for each draw;
      (2) estimate the model evidence logZ by prior Monte Carlo, so the wet and
          dry runs can be compared via P(wet) = Z_wet/(Z_wet+Z_dry).
    The triple-quoted blocks below are templates with __PLACEHOLDERS__ that get
    substituted at the end; underscore-prefixed names (_np, _th, ...) avoid
    clobbering any variables already living in the notebook's namespace.
    """
    swcol = th["swcol"]
    # ---- common preamble: dump chain + compute binary thickness samples ----
    # _th = 8500 m crustal column * S_w * porosity(theta[:,1]).  For the dry run
    # _SW = 0 so every thickness is exactly 0 km, which is the whole point.
    common = """
# --- R1.7 two-end-member tail (__EM__, S_w=__SW__) ---
import os as _os, numpy as _np
_SW = __SW__
_em = "__EM__"
_chain = sampler.get_chain(flat=True)
_os.makedirs(output_dir, exist_ok=True)
_np.save(_os.path.join(output_dir, f'samples_{case_name}.npy'), _chain)
_th = 8500.0 * _SW * _chain[:, 1]              # binary thickness [m]; dry -> 0
_np.save(_os.path.join(output_dir, f'thickness_samples_{case_name}.npy'), _th)
_rng = _np.random.default_rng(12345)
_N = __EVN__
"""
    # ---- evidence body: draw _N points from the prior box, evaluate the
    #      log-LIKELIHOOD at each, and average exp(loglike) -> Z (prior MC).
    #      Two flavours depending on how the notebook exposes its likelihood:
    #        "ds"  -> call log_post(theta, lb, ub, d, s, H); reuse notebook lb/ub
    #        "obs" -> call log_probability(x); we hard-code the prior box pb_lb/ub
    #      In both, the S_w column is pinned to 0.5 (an interior value) only so
    #      the draw passes the bound check -- the forward model ignores it and
    #      uses the fixed _SW we spliced in above.
    if th["style"] == "ds":
        body = """_lo, _hi = _np.asarray(lb, float), _np.asarray(ub, float)
_dr = _rng.uniform(_lo, _hi, size=(_N, _lo.size)); _dr[:, __SWCOL__] = 0.5  # interior: passes bound check; forward uses fixed _SW
_ll = _np.full(_N, -_np.inf)
for _i in range(_N):
    try: _ll[_i] = log_post(_dr[_i], lb, ub, d, s, H)[0]
    except Exception: _ll[_i] = -_np.inf
"""
    else:
        body = """_lo = _np.array(__PBLB__); _hi = _np.array(__PBUB__)
_dr = _rng.uniform(_lo, _hi, size=(_N, _lo.size)); _dr[:, __SWCOL__] = 0.5  # interior: passes bound check; forward uses fixed _SW
_ll = _np.full(_N, -_np.inf)
for _i in range(_N):
    try: _ll[_i] = log_probability(_dr[_i])[0]
    except Exception: _ll[_i] = -_np.inf
"""
    # ---- finalize: stable log-mean-exp of the loglikes -> logZ, then save.
    #      _mx shift avoids overflow; if no draw was finite, logZ = -inf.
    fin = """_f = _np.isfinite(_ll)
if _f.any():
    _mx = _ll[_f].max(); _logZ = _mx + _np.log(_np.mean(_np.exp(_ll - _mx)))
else:
    _logZ = float('-inf')
open(_os.path.join(output_dir, f'logZ_{case_name}.txt'), 'w').write(repr(float(_logZ)))
print(f'>> [BINARY {_em}] {case_name}: logZ={_logZ:.3f} thick_mean_km={_th.mean()/1000:.4f} n={_chain.shape[0]}')
"""
    # Splice the per-theory / per-end-member values into the templates.
    out = common + body + fin
    out = (out.replace("__EM__", em).replace("__SW__", sw_literal)
              .replace("__EVN__", str(evn)).replace("__SWCOL__", str(swcol)))
    if th["style"] == "obs":
        out = out.replace("__PBLB__", th["pb_lb"]).replace("__PBUB__", th["pb_ub"])
    return out


def make(th):
    """Generate the wet AND dry notebook + runner for one theory `th`."""
    for em, sw in EM.items():                       # em in {"wet","dry"}, sw in {"1.0","0.0"}
        evn = EVN_CLI or th["evn"]                  # CLI override wins, else per-theory default
        # 1) Load the theory's pristine notebook.
        nb = nbformat.read(os.path.join(ROOT, th["d"], th["nb"]), as_version=4)
        old, new_t = th["repl"]; new = new_t.format(SW=sw)   # build the S_w-fixing replacement
        napp = 0                                    # count how many cells we patched (sanity check)
        # 2) Walk every code cell and apply the edits.
        for c in nb.cells:
            if c.cell_type != "code":
                continue
            s = c.source
            if old in s:                            # FIX S_w: overwrite the sampled value
                s = s.replace(old, new); napp += 1
            if NSTEPS is not None:                  # smoke-test mode: shrink chain length
                s = re.sub(r"(?m)^(\s*)Nsteps\s*=\s*\d+", rf"\1Nsteps = {NSTEPS}", s)
                s = re.sub(r"(?m)^(\s*)nsteps\s*=\s*\d+", rf"\1nsteps = {NSTEPS}", s)
            c.source = s
        # 3) Append our results-computing cell (chain dump + thickness + evidence).
        nb.cells.append(nbformat.v4.new_code_cell(tail(th, sw, em, evn)))
        stem = os.path.splitext(th["nb"])[0]
        nbformat.write(nb, os.path.join(ROOT, th["d"], f"{stem}_binary_{em}.ipynb"))

        # 4) Clone the theory's own run.py, just retargeting it at the new
        #    notebook + a dedicated outputs_binary_<em>/ dir. We do NOT rewrite
        #    the runner's logic (it still does the per-case data patching);
        #    we only swap paths and force allow_errors=True so one bad cell
        #    doesn't abort the whole batch.
        txt = open(os.path.join(ROOT, th["d"], th["run"])).read()
        txt = re.sub(r'orig_path\s*=\s*"[^"]*"', f'orig_path = "{stem}_binary_{em}.ipynb"', txt)
        txt = re.sub(r'output_dir\s*=\s*"[^"]*"', f'output_dir = "outputs_binary_{em}"', txt)
        txt = re.sub(r'out_tpl\s*=\s*"[^"]*"', f'out_tpl    = "binary_{em}_executed_{{name}}.ipynb"', txt)
        txt = re.sub(r'NotebookClient\(([^)]*)\)',
                     lambda m: f'NotebookClient({m.group(1)}, allow_errors=True)'
                     if 'allow_errors' not in m.group(1) else m.group(0), txt, count=1)
        open(os.path.join(ROOT, th["d"], f"run_binary_{em}.py"), "w").write(txt)
        # Report: repl= should be >=1, otherwise the S_w fix silently didn't apply!
        print(f"{th['key']:14} {em:3} repl={napp} evn={evn} -> {th['d']}/{stem}_binary_{em}.ipynb")


if __name__ == "__main__":
    # Generate all 16 notebooks (8 theories x {wet,dry}). The 48 actual runs
    # (x3 cases) are launched separately by launch_all.sh.
    print(f"NSTEPS={NSTEPS} EVN_CLI={EVN_CLI}")
    for th in THEORIES:
        make(th)
