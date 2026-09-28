# Figure S4

Reproducible code and outputs for Figure S4: sensitivity to first-order versus saturable Michaelis-Menten PDE clearance.

## Run

```bash
python figS04_simulation.py
```

Outputs are written to `outputs/`. `figS04_parameters_reference.json` records the frozen corrected parameter set.

## Notes

- All transport and numerical settings inherit the final Figure 2 implementation; only the PDE sink law changes.
- Michaelis-Menten degradation is matched to the first-order sink at `c_max = sqrt(K_A*K_B)` via `Vmax = k_ref*(K_M + c_max)`.
- Panels A-C use the final post-pulse reflecting boundary treatment; panel D uses the corrected Figure 2E-F 5 nM baseline and half-cell flux boundaries.
- Penetration depth uses the same linearly interpolated 10%-of-overall-peak threshold as Figure 2.
