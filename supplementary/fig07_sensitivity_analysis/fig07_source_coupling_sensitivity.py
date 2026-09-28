#!/usr/bin/env python3
"""Figure 7 sensitivity to source-to-path coupling distance.

The localized AC disk source, source radius, turnover, downstream transport,
RI activation, clearance, and numerical parameters are held at the Figure 7
reference values. Only the distance at which the downstream organized-path
model samples the source-proximal disk field is varied.

Output:
  outputs/Fig7_source_coupling_sensitivity.csv
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

COUPLING_DISTANCES_UM = [0.01, 0.02, 0.03, 0.05, 0.10, 0.20, 0.50, 1.00]
TARGET_UM = 20.0


def main() -> None:
    with PARAMS.open("r", encoding="utf-8") as fh:
        p0 = json.load(fh)

    rows = []
    for zc in COUPLING_DISTANCES_UM:
        p = copy.deepcopy(p0)
        p["geometry"]["common_coupling_distance_um"] = float(zc)
        t_eval = f7.make_time_grid(p)
        c_input = float(f7.common_source_input(p))
        org = f7.simulate_organized(p, t_eval, c_input)
        i20 = f7.nearest_index(org["x"], TARGET_UM)
        peak_release = float(np.max(org["release"][i20]))
        rows.append({
            "coupling_distance_um": float(zc),
            "coupling_distance_nm": 1000.0 * float(zc),
            "common_input_uM": c_input,
            "common_input_nM": 1000.0 * c_input,
            "organized_peak_local_PKAc_20um_uM_per_s": peak_release,
            "organized_peak_local_PKAc_20um_over_basal": peak_release / org["R_rest"],
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "Fig7_source_coupling_sensitivity.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
