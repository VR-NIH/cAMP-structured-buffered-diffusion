# Figure 6 - Organized versus conventional buffered transport limits

This directory generates Figure 6, a two-panel comparison of effective cAMP transport in organized RI and conventional unorganized buffering limits.

## Files

- `fig06_simulation.py` - canonical figure-generation script.
- `fig06_parameters_reference.json` - reference parameter summary for the plotted surfaces.
- `outputs/` - generated figure files and parameter/summary records.

## Model summary

### Panel A: organized RI

The organized transport surface combines the cAMP occupancy term

`[c/(K_B + c)] [K_A/(K_A + c)]`

with an illustrative architecture factor

`g_R(rho_R) = 4 rho_R / (1 + rho_R)^2`,

where `rho_R = R_T/R_*` is a dimensionless relative RI abundance. The displayed surface is normalized to its own maximum. `R_*` is an illustrative reference abundance; no absolute RI concentration optimum is inferred from Figure 5.

With `K_B = 0.1 uM` and `K_A = 1.0 uM`, the cAMP-dependent maximum occurs at

`c = sqrt(K_A K_B) = 0.3162 uM`,

and the architecture factor is maximal at `rho_R = 1`.

### Panel B: conventional buffered diffusion

The conventional panel uses the rapid-equilibrium immobile-buffer limit with

`K_D,eff = sqrt(K_A K_B) = 0.3162 uM`,

`beta = B_T/K_D,eff`,

`kappa = beta/(1 + c/K_D,eff)^2`,

and

`D_buffer/D_free = 1/(1 + kappa)`.

The two panels represent different limiting effective transport regimes. Their colorbars have separate meanings and the two surfaces are not added pointwise.

## Plot ranges

- cAMP: `0.01-10 uM`
- Relative organized RI abundance, `rho_R`: `0.01-100`
- Conventional dimensionless buffer abundance, `beta`: `0.01-100`
- Each plotted axis uses 320 logarithmically spaced points.

## Run

From the repository root:

```bash
python fig06/fig06_simulation.py
```

or from this directory:

```bash
python fig06_simulation.py
```

Required Python packages are `numpy` and `matplotlib`.

## Outputs

The script writes the following files to `outputs/`:

- `fig06.pdf`
- `fig06.png`
- `fig06_parameters.json`
- `fig06_parameters.txt`
- `fig06_summary.txt`

The canonical script contains the final two-panel layout, including wrapped y-axis labels and explicit inter-panel spacing. No separate layout-fix script is required.
