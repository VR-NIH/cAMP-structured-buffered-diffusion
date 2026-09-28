# Figure 2

Reproducible code and outputs for the revised Figure 2: occupancy-dependent RI mobility, PDE range control, and equal-input temporal packaging.

## Run

```bash
python fig02_simulation.py
```

The script reads `fig02_parameters.json` and writes the figure and metrics to `outputs/`.

## Notes

- Panels A-D use a 1-s 2.5 uM proximal concentration clamp, followed by a reflecting proximal boundary; the distal boundary is reflecting throughout.
- Panels E-F use a fixed proximal flux with half-cell finite-volume boundary corrections and a 5 nM resting cAMP concentration.
- `D_relay(c)` is an effective structured-buffered mobility, not the Brownian diffusion coefficient of free cAMP.
- Penetration depth uses a 10%-of-overall-peak threshold with linear interpolation between the two bracketing spatial nodes.
- The script also writes the local log-log clearance slope and a protocol-matched constant-mobility control with `D_const = D_relay(c_max)`.
