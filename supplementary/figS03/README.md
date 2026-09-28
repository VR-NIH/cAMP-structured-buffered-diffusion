# Figure S3

Reproducible code and outputs for Figure S3: sensitivity of equal-input pulse fractionation to effective PDE-mediated clearance and total cAMP input.

## Run

```bash
python figS03_simulation.py
```

Outputs are written to `outputs/`. `figS03_parameters_reference.txt` records the frozen parameter set.

## Notes

- This implementation matches the corrected Figure 2E-F flux boundary treatment and 5 nM resting cAMP condition.
- Distal mobility exposure is `E_relay = integral D_relay(c_distal(t)) dt`.
- Pulse-number optima are interpreted as broad regime-dependent responses, not universal preferred frequencies.

For maximum cross-platform reproducibility the packaged script defaults to `N_JOBS = 1`; this may be increased for local parallel execution without changing model output.
