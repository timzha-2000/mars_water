# Mars mid-crust rock-physics inversion

This repository contains all the rock-physics model implementations and analysis
code for the study:

**"Rock physics modeling reveals large uncertainty in midcrustal liquid water on Mars"**
*Mufan Zha, Per Avseth, Paul Sava*

We test eight rock-physical theory configurations across three seismic parameter
sets to evaluate the sensitivity of inferred liquid water distribution in Mars'
mid-crust. Each model lives in its own subdirectory under `models/`.

## Repository layout

```
models/      one subdirectory per rock-physics configuration (notebook + run.py)
plotting/    scripts that build the figures in the paper
analysis/    convergence, binary-saturation, and across-theory spread analyses
wmm24/       independent reimplementation of Wright et al. (2024) (MATLAB + Python)
figures/     example figure outputs
```

All scripts resolve paths relative to the repository root, so the repository runs
from a fresh clone without editing any absolute paths.

## Models

| # | Directory | Dry frame | Fluid mixing | Fluid substitution |
|---|-----------|-----------|--------------|-------------------|
| 1 | `models/1_SCM_Voigt` | Berryman's SCM | Voigt | Gassmann-Biot |
| 2 | `models/2_SCM_Hill`  | Berryman's SCM | Hill | Gassmann-Biot |
| 3 | `models/3_HS_Voigt`  | Hashin-Shtrikman | Voigt | Gassmann-Biot |
| 4 | `models/4_HS_Hill`   | Hashin-Shtrikman | Hill | Gassmann-Biot |
| 5 | `models/5_DEM_Voigt` | Differential Effective Medium | Voigt | -- |
| 6 | `models/6_DEM_Hill`  | Differential Effective Medium | Hill | -- |
| 7 | `models/7_KT`        | Kuster-Toksoz | -- | -- |
| 8 | `models/8_VRH`       | Voigt-Reuss-Hill | -- | -- |

All inversions use 50,000 MCMC iterations via the `emcee` affine-invariant
ensemble sampler (the same sampler used by Wright et al. 2024), with
`n_walkers = 3 * n_dim`.

## Three seismic parameter sets

The velocity case with $V_p = 4.1$ km/s, $V_s = 2.5$ km/s corresponds to the
parameters used in Wright et al. (2024); see Table 2 in the paper. The three
cases are tagged `A_constraints_away` (3.8), `B_wright_inherited` (4.1), and
`C_insight_marsquake` (4.7).

## Running the continuous-saturation inversions

Each model subdirectory contains a notebook (the forward model + inversion) and a
`run.py` that executes all three velocity cases headlessly, writing chains to a
local `outputs_*/` directory:

```bash
cd models/1_SCM_Voigt
python run.py                      # all three cases
python run.py A_constraints_away   # a single case
```

## Plotting (paper figures)

After the inversions have been run, the `plotting/` scripts build the figures:

| Script | Figure |
|---|---|
| `make_thickness_barplot.py`            | mode/median/mean thickness barplot (Fig. 1) |
| `make_thickness_ridgeline.py`          | continuous thickness ridgeline (Fig. 2) |
| `make_scm_vp41_interpretation.py`      | SCM (phi, S_w) crossplot: geometric origin of the thickness posterior (Fig. 3) |
| `make_porosity_ridgeline.py`           | porosity marginals (App A) |
| `make_saturation_ridgeline.py`         | saturation marginals (App A) |
| `make_thickness_overlay.py`            | thickness overlay panel |
| `make_thickness_ridgeline_binary.py`   | binary wet-branch thickness ridgeline (App B) |
| `make_thickness_barplot_binary.py`     | binary thickness barplot (App B) |
| `make_prior_sensitivity_ridgeline.py`  | prior-box sensitivity (App D) |

```bash
python plotting/make_thickness_ridgeline.py
```

## Analysis (`analysis/`)

| Script | Purpose |
|---|---|
| `compute_convergence.py` | MCMC convergence diagnostics — integrated autocorrelation time, N_steps/tau, and ESS for all 24 theory–velocity combinations (Appendix C). Reproduces Table A.3. |
| `setup_binary.py`        | Generates the binary (end-member S_w in {0,1}) notebooks and `run_binary_{wet,dry}.py` runners in each model directory, following Wright et al. (2025) (Appendix B). |
| `launch_all.sh`          | Convenience launcher for all 48 binary runs (8 models × 3 cases × {wet, dry}). |
| `make_binary_table.py`   | Builds the binary wet/dry probability and thickness table (Table A.2). |
| `spread_analysis.py`     | Across-theory disagreement vs. Vp/Vs, by range and coefficient of variation (Table 4). |
| `run_hotstart.py`        | Warm/hot-start re-run of a single inversion: reuses the model's executed notebook (the exact forward model + sampler), applies Wright et al.'s cold-start → RMSE<3 resample → production protocol, and saves the resulting thickness posterior. For the Appendix C initialization-robustness check. |
| `launch_hotstart_all.sh` | Runs `run_hotstart.py` over all 24 theory–velocity combinations → `analysis/hotstart_outputs/`. Requires each model's `run.py` to have been executed first (uses the per-case `*_executed_*.ipynb` notebooks). |
| `compare_hotstart.py`    | Tabulates cold-start vs. warm-start median / mean / P(h<0.5) for all 24 combinations and the worst-case shift — the Appendix C warm-start comparison. |

Convergence (after running the inversions):
```bash
python analysis/compute_convergence.py
```

Binary-saturation analysis:
```bash
python analysis/setup_binary.py     # generate binary runners in each model dir
bash analysis/launch_all.sh         # run all 48 (or run each models/<m>/run_binary_*.py)
python analysis/make_binary_table.py
python plotting/make_thickness_ridgeline_binary.py
```

Warm/hot-start robustness check (Appendix C; after the continuous inversions are run):
```bash
bash analysis/launch_hotstart_all.sh   # 24 warm-start re-runs -> analysis/hotstart_outputs/
python analysis/compare_hotstart.py    # cold-start vs warm-start comparison table
```

## Wright et al. (2024) reimplementation (`wmm24/`)

An independent reimplementation of the Wright et al. (2024) Berryman-SCM
inversion, used to reproduce their result (Appendix A) and to run the
prior-box sensitivity study (Appendix D, prior Sets 2 and 3). Both a MATLAB
version (`wmm24/matlab/`, the same affine-invariant ensemble sampler as the
original) and a Python port (`wmm24/python/`) are provided. The Appendix-D
prior-box runs are produced by `RunWiderCont.m`, `RunReplyCont.m`,
`RunWiderBinary.m`, and `RunReplyBinary.m`.

## Figures

Example outputs are saved under `figures/`.

## References

- Wright, V., Morzfeld, M., & Manga, M. (2024). Liquid water in the Martian mid-crust. *PNAS*, 121, e2409983121.
- Wright, V., Morzfeld, M., & Manga, M. (2025). Reply to Xiao et al. *PNAS*, 122, e2505168122.
- Xiao, W., Pan, L., Wang, Y., & Li, J. (2025). Liquid water might not be the only answer. *PNAS*, 122, e2503071122.
- Goodman, J., & Weare, J. (2010). Ensemble samplers with affine invariance. *Comm. Appl. Math. Comput. Sci.*, 5(1), 65–80.
- Mavko, G., Mukerji, T., & Dvorkin, J. (2020). *The Rock Physics Handbook*, 3rd Ed. Cambridge University Press.
