# Figure S2 - occupancy-threshold sensitivity

This package tests whether the non-monotonic structured-buffered mobility window depends on the baseline effective occupancy thresholds.

The script plots normalized `D_relay(c)` for representative `K_B`/`K_A` pairs and maps the analytic peak position

`c_max = sqrt(K_A K_B)`

over a logarithmic threshold grid. `K_B` and `K_A` are effective occupancy thresholds in the reduced model, not uniquely assigned microscopic dissociation constants.

Run:

```bash
python figS02_threshold_sensitivity.py
```

Outputs are written to `outputs/` as PNG, PDF, and a parameter summary.
