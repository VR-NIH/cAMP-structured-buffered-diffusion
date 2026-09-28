# Figure 5

RI-abundance/output-capacity scan used for Figure 5.

## Run

```bash
python fig05_simulation.py
```

The script writes reproducibility outputs to `outputs/`.

## Model notes

- `R_T` is scanned from 0.1 to 100 uM.
- `R_T` does not directly rescale `D_relay(c)`.
- Holoenzyme-associated PKAc-output capacity is `H_T = R_T / rho_RC` with `rho_RC = 4`.
- Figure 5 includes cumulative local release (`Q_rel`) so that `H_intact = max(H_T - Q_rel, 0)`.
- The causal control sets induced PDE feedback to zero (`k_PDE_i = 0`) while retaining basal cAMP clearance.
- cAMP and PKAc use conservative half-cell boundary corrections; `D_relay` uses harmonic interface averaging.

## Included outputs

- `fig05.png` and `fig05.pdf`: final figure
- `fig05_metrics.csv`: full RI-abundance scan
- `fig05_summary.json`: key numerical summaries
- `fig05_parameters.txt`: parameter and implementation record

`fig05_parameters_reference.json` is the frozen parameter snapshot corresponding to the archived figure.
