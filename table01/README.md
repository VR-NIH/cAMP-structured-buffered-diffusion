# Table 1 - strong-input transport comparison

`table01_high_input_control.py` reproduces the Table 1 comparison under a 1-s, 2.5-uM proximal free-cAMP pulse using the Figure 7 downstream parameters.

The three branches are:

- conventional radial buffered diffusion;
- radial three-dimensional diffusion with conventional buffer retardation removed and `D = D_free = 100 um^2/s`;
- organized-path structured transport.

The no-buffer radial branch tests whether removing conventional buffer retardation and allowing the full free-diffusion coefficient are sufficient to reproduce the organized response. The comparison does **not** isolate geometry as a single independent variable because the organized branch also retains its occupancy-dependent structured-mobility law and reduced productive coordinate.

## Outputs

- `outputs/Table1_high_input_transport_comparison.csv` - manuscript-facing Table 1 quantities.
- `outputs/Table1_high_input_diagnostics.csv` - source-equivalent calibration, timing, basal output, proximal mobility, and formal-output diagnostics.

The manuscript-facing columns are:

- transport condition;
- maximum local mobility scale over 0-20 um;
- peak total cAMP at 20 um;
- stimulus-evoked peak cAMP at 20 um;
- peak local PKAc release relative to basal;
- absolute peak local PKAc release.

## Run

From repository root:

```bash
python table01/table01_high_input_control.py
```
