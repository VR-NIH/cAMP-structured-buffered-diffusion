# Figure 1C - occupancy-dependent mobility window

Standalone code for the quantitative panel of Figure 1.

The script plots the structured-buffered mobility

`D_relay(c) = D_relay,0 [c/(K_B+c)] [K_A/(K_A+c)]`

for the baseline threshold pair (`K_B = 0.1 uM`, `K_A = 1 uM`) and the equal-threshold control (`K_B = K_A = 0.1 uM`). The analytic maximum is `c_max = sqrt(K_A K_B)`.

Run:

```bash
python fig01C_mobility_window.py
```

Outputs are written to `outputs/` as PNG, PDF, metrics CSV, and parameter JSON. `D_relay,0` is an effective organized-path mobility prefactor, not the Brownian diffusion coefficient of free cAMP.
