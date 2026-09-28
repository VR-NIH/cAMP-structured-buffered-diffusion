#!/usr/bin/env python3
"""Figure 7 sensitivity to localized AC source strength.

The source radius, 20-nm coupling distance, resting cAMP, downstream organized
transport, RI activation, clearance, and numerical parameters are held at the
Figure 7 reference values. AC turnover is kept at the reference 100 cAMP/s per
active AC and the active-AC count is varied to produce 1,000-10,000 cAMP/s.

Output:
  outputs/Fig7_source_strength_sensitivity.csv
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

import fig07_simulation as f7

HERE = Path(__file__).resolve().parent
PARAMS = HERE / "fig07_parameters.json"
OUT = HERE / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

SOURCE_RATES_MOLECULES_PER_S = [1000.0, 2500.0, 5000.0, 10000.0]
TARGET_UM = 20.0


def main() -> None:
    with PARAMS.open("r", encoding="utf-8") as fh:
        p0 = json.load(fh)

    turnover = float(p0["source_biology"]["turnover_cAMP_per_AC_per_s"])
    rows = []
    for qsrc in SOURCE_RATES_MOLECULES_PER_S:
        p = copy.deepcopy(p0)
        active_ac = qsrc / turnover
        p["source_biology"]["active_AC_count"] = float(active_ac)
        p["source_biology"]["total_cAMP_production_molecules_per_s"] = float(qsrc)

        t_eval = f7.make_time_grid(p)
        c_surface = float(f7.source_surface_concentration(p))
        c_input = float(f7.common_source_input(p))
        org = f7.simulate_organized(p, t_eval, c_input)
        i20 = f7.nearest_index(org["x"], TARGET_UM)
        peak_release = float(np.max(org["release"][i20]))

        rows.append({
            "source_rate_molecules_per_s": float(qsrc),
            "equivalent_active_AC_at_reference_turnover": float(active_ac),
            "source_surface_cAMP_uM": c_surface,
            "source_surface_cAMP_nM": 1000.0 * c_surface,
            "common_input_20nm_uM": c_input,
            "common_input_20nm_nM": 1000.0 * c_input,
            "organized_peak_local_PKAc_20um_uM_per_s": peak_release,
            "organized_peak_local_PKAc_20um_over_basal": peak_release / org["R_rest"],
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "Fig7_source_strength_sensitivity.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
