# Structured buffered diffusion of cAMP

This repository contains the simulation and analysis code used to generate the
main and supplementary figures for the accompanying manuscript on structured
buffered diffusion and cAMP signaling.

## Repository structure

- `fig01C/` — Figure 1C calculations
- `fig02/` — Figure 2 simulations
- `fig03/` — Figure 3 PKAc-dependent PDE feedback simulations
- `fig04/` — Figure 4 parameter-sweep calculations
- `fig05/` — Figure 5 RI-abundance scan
- `fig06/` — Figure 6 organized versus conventional buffered-diffusion limits
- `fig07/` — Figure 7 organized versus unorganized RI transport comparison
- `supplementary/` — supplementary-figure simulations and analyses

Individual figure directories contain the corresponding simulation or analysis
script, parameter files, figure-specific documentation where applicable, and an
`outputs/` directory containing numerical results and generated figures.

## Software requirements

The calculations were performed in Python using:

- Python [3.14.3]
- NumPy 2.4.4
- SciPy 1.17.1
- pandas 3.0.3
- Matplotlib 3.10.8
- joblib 1.5.3

Exact Python package versions are listed in `requirements.txt`.

To install the required packages:

```bash
python -m pip install -r requirements.txt