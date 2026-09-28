# Figure S5

Sensitivity of distal free-PKAc signaling to the structured-buffered mobility prefactor `D_relay,0`.

## Run

```bash
python figS05_simulation.py
```

The script writes all generated files to `outputs/`.

## Comparison

The canonical Figure 3 paired-pulse model is repeated with:

- `D_relay,0 = 50 um^2/s`
- `D_relay,0 = 10 um^2/s`

All other model, source, feedback, boundary, and numerical parameters are unchanged. Free PKAc is evaluated at `s = 20 um`.

`figS05_metrics.csv` reports the first distal free-PKAc peak, the second local maximum after the second input, their times, and the 0-18 s free-PKAc AUC. The second local maximum is searched over 6-12 s to distinguish it from residual PKAc present at second-pulse onset.

`D_relay,0` is an effective organized-path mobility prefactor and is not the Brownian diffusion coefficient of free cAMP.
