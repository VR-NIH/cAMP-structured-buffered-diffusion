# Figure 7 sensitivity analyses

These scripts use the Figure 7 reference source, geometry, RI activation, resting cAMP, clearance, and numerical formulation unless the named quantity is explicitly varied. They are reproducibility analyses for the sensitivity calculations reported with Figure 7 and in the Appendix.

## Source-to-path coupling distance

`fig07_source_coupling_sensitivity.py` varies only the distance at which the organized-path model samples the localized AC disk field: 10, 20, 30, 50, 100, 200, 500, and 1000 nm. AC production and all downstream parameters remain fixed.

Output:

- `outputs/Fig7_source_coupling_sensitivity.csv`

## AC source strength

`fig07_source_strength_sensitivity.py` varies localized AC production over 1,000, 2,500, 5,000, and 10,000 cAMP molecules/s while holding the source radius, 20-nm coupling distance, resting cAMP, and downstream parameters fixed. With the reference turnover of 100 cAMP molecules per active AC per second, these rates correspond to 10, 25, 50, and 100 active AC molecules.

Output:

- `outputs/Fig7_source_strength_sensitivity.csv`

## Structured-mobility scale

`fig07_Drelay_sensitivity.py` varies only `D_relay,0` over 10, 20, 50, and 100 um^2/s. The sensitivity observation window is extended to 30 s so delayed low-mobility peaks are not truncated.

Outputs:

- `outputs/Fig7_Drelay_sensitivity.csv`
- `outputs/Fig7_Drelay_sensitivity.png`
- `outputs/Fig7_Drelay_sensitivity.pdf`

## Lateral escape

`fig07_escape_sensitivity.py` adds a phenomenological loss term

`-k_escape * (c - c_rest)`

that removes evoked cAMP from the reduced productive coordinate. This term is distinct from PDE-mediated chemical degradation. Escaped cAMP is not transferred to an explicit bulk compartment and is not allowed to re-enter the productive coordinate. `k_escape = 0` is the reference high-recapture organized-path limit.

The observation window is 30 s. Reported values are supplemented with a denser grid for monotonic curves and interpolation of output thresholds.

Outputs:

- `outputs/Fig7_escape_sensitivity.csv`
- `outputs/Fig7_escape_thresholds.csv`
- `outputs/Fig7_escape_sensitivity.png`
- `outputs/Fig7_escape_sensitivity.pdf`

## Run

From repository root:

```bash
python fig07/fig07_source_coupling_sensitivity.py
python fig07/fig07_source_strength_sensitivity.py
python fig07/fig07_Drelay_sensitivity.py
python fig07/fig07_escape_sensitivity.py
```
