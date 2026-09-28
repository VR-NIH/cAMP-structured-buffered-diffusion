# Figure S6 - Low-R_T limiting paired-pulse facilitation

This folder contains the reproducible simulation code, parameters, numerical outputs, and final supplementary figure for Figure S6.

## Run

```bash
python figS06_simulation.py
```

The script writes outputs to `outputs/`.

## Canonical condition

- `R_T = 0.2 uM`
- `K_A = 1 uM`
- `k_off,A = 80 s^-1`
- `k_on,A = 80 uM^-1 s^-1`
- 4-s interpulse interval, defined from the end of P1 to the onset of P2
- distal readout at `s = 20 um`

This is the open-star condition in Fig. 4D. The displayed ratio is the paired-pulse P2 peak divided by the peak from a matched isolated P1 simulation (`P2/P1,iso`). Output capacity is fixed during the paired-pulse simulation; cumulative holoenzyme depletion is not included.
