# Figure S7 - source amplitude x lateral escape

Figure S7 examines how proximal cAMP input and loss from the reduced productive coordinate jointly determine distal organized-path signaling.

Every grid point is a full organized-path simulation using the Figure 7 reference spatial grid, 15-s observation window, RI activation kinetics, resting cAMP, structured mobility, and PDE-mediated clearance. Only two quantities are varied:

1. proximal free-cAMP concentration during the 1-s input;
2. phenomenological lateral escape rate `k_escape`.

Lateral escape removes only the evoked component from the productive coordinate and does not return escaped cAMP to that coordinate.

## Outputs

- `outputs/FigS7_source_escape_grid.csv` - complete long-form numerical grid.
- `outputs/FigS7_source_escape_representative.csv` - reference, 500-nM, and 1000-nM representative curves.
- `outputs/FigS7_source_escape_monotonicity.csv` - monotonicity audit across escape at each source amplitude.
- `outputs/FigS7_source_escape_map.npz` - compact matrix archive corresponding to the CSV grid.
- `outputs/FigS7_source_escape_map_final.png`
- `outputs/FigS7_source_escape_map_final.pdf`

## Run

From repository root:

```bash
python figS07/figS07_source_escape_map.py
```

The script imports `fig07_simulation.py`, `fig07_parameters.json`, and the organized-path escape implementation from `fig07/`.
