# Figure 3

Canonical paired-pulse structured-buffered cAMP / PKA / PDE-feedback simulation used for Figure 3.

## Run

```bash
python fig03_simulation.py
```

The script writes all generated files to `outputs/`.

## Main model features

- Structured-buffered cAMP mobility: `D_relay(c)`
- Dynamic RI activation-state fraction `f_AB`
- Fixed holoenzyme-associated PKAc-output capacity during the paired-pulse simulation
- Diffusible free PKAc
- Local nondiffusing `PDE*` feedback state
- Canonical feedback condition and `k_PDE_i = 0` causal control
- Two 1-s proximal flux inputs at 0-1 s and 5-6 s
- Distal local-release readout at 20 um

`fig03_metrics.csv` contains the canonical and no-induced-feedback response metrics. The reported peak and integrated local-release P2/P1 ratios are computed over 0-5 s and 5-10 s.
