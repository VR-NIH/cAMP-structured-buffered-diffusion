# Figure 4 - Paired-pulse phase maps

This folder contains the reproducible simulation code, parameters, numerical outputs, and final figure for Figure 4.

## Run

```bash
python fig04_simulation.py
```

The script writes outputs to `outputs/`.

## Model notes

- Structured-buffered cAMP transport uses occupancy-dependent `D_relay(c)`.
- Paired-pulse simulations use fixed PKAc-output capacity `H_T = R_T/rho_RC`; cumulative holoenzyme depletion is not included.
- Free PKAc activates a local nondiffusing PDE-feedback state.
- `R_T` does not directly rescale `D_relay`.
- Top-row maps report peak P2 divided by the peak of a matched isolated P1 simulation.
- Bottom-row maps report integrated local distal PKAc release from the matched isolated P1 simulation over 0-5 s.
- The open star corresponds to the low-R_T limiting facilitation example shown in Fig. S6.

The full phase-map calculation is computationally intensive because it evaluates four 16 x 16 parameter sweeps plus matched isolated-pulse references.
