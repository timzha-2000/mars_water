# Mars

This repository contains all the rock-physics model implementations for the study:

**"Rock physics modeling reveals large uncertainty in midcrustal liquid water on Mars"**
*Mufan Zha, Per Avseth, Paul Sava*

We test eight rock-physical theory configurations across three seismic parameter sets to evaluate the sensitivity of inferred liquid water distribution in Mars' mid-crust. Each model is contained in its own subdirectory under `models/`.

## Models

| # | Directory | Dry frame | Fluid mixing | Fluid substitution |
|---|-----------|-----------|--------------|-------------------|
| 1 | `1_SCM_Voigt` | Berryman's SCM | Voigt | Gassmann-Biot |
| 2 | `2_SCM_Hill` | Berryman's SCM | Hill | Gassmann-Biot |
| 3 | `3_HS_Voigt` | Hashin-Shtrikman | Voigt | Gassmann-Biot |
| 4 | `4_HS_Hill` | Hashin-Shtrikman | Hill | Gassmann-Biot |
| 5 | `5_DEM_Voigt` | Differential Effective Medium | Voigt | -- |
| 6 | `6_DEM_Hill` | Differential Effective Medium | Hill | -- |
| 7 | `7_KT` | Kuster-Toksoz | -- | -- |
| 8 | `8_VRH` | Voigt-Reuss-Hill | -- | -- |

All inversions use 50,000 MCMC iterations via the `emcee` ensemble sampler.

## Three seismic parameter sets

The velocity case with $V_p = 4.1$ km/s, $V_s = 2.5$ km/s corresponds to the parameters used in Wright et al. (2024). See Table 2 in the paper for details.

## Usage

Each model subdirectory contains:
- A Jupyter notebook (`.ipynb`) with the forward model and inversion
- A `run.py` script that executes all three velocity cases headlessly

To run a single model (e.g., SCM + Voigt):
```bash
cd models/1_SCM_Voigt
python run.py
```

To run a specific velocity case:
```bash
python run.py A_constraints_away
```

The `plotting/` directory contains scripts to generate the summary figures from the paper.

## Figures

The outputs of the codes are saved under /figures.

## References

- Wright, V., Morzfeld, M., & Manga, M. (2024). Liquid water in the Martian mid-crust. *PNAS*, 121, e2409983121.
- Mavko, G., Mukerji, T., & Dvorkin, J. (2020). *The Rock Physics Handbook*, 3rd Ed. Cambridge University Press.
