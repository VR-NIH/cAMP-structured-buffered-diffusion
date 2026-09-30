#!/usr/bin/env python3
"""Sensitivity of Fig. 6 organized-path output to D_relay,0.

Uses the reference Fig. 6 source, geometry, kinetics, resting state, numerical
settings, with a 30-s sensitivity-analysis observation window. Only D_relay,0 is varied in the organized
branch. The longer window ensures delayed low-mobility responses reach a true peak. The conventional branch is run once because it does not depend on
D_relay,0.

Outputs:
  outputs/Fig6_Drelay_sensitivity.csv
  outputs/Fig6_Drelay_sensitivity.png
  outputs/Fig6_Drelay_sensitivity.pdf
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import linregress
import matplotlib.pyplot as plt

import fig06_simulation as f6

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)
PARAMS = HERE / "fig06_parameters.json"

# Multiples of the reference 50 um^2/s value: 0.2x, 0.4x, 1x, 2x.
D_VALUES = [10.0, 20.0, 50.0, 100.0]
TARGET_UM = 20.0
SENSITIVITY_T_END_S = 30.0


def fit_attenuation(metrics: pd.DataFrame) -> tuple[float, float]:
    d = metrics.sort_values("distance_um")
    slope, intercept, r, _, _ = linregress(
        d["distance_um"].to_numpy(),
        np.log(d["peak_evoked_PKAc_release_uM_per_s"].to_numpy()),
    )
    return -1.0 / slope, r * r


def main() -> None:
    with PARAMS.open("r", encoding="utf-8") as fh:
        p0 = json.load(fh)

    # Extend the observation window for the mobility sweep while preserving
    # approximately the reference post-pulse output-sampling interval.
    reference_tend = float(p0["time"]["t_end_s"])
    reference_nlate = int(p0["time"]["late_n_points"])
    t_switch = float(p0["time"]["stimulus_start_s"] + p0["time"]["stimulus_duration_s"])
    reference_late_dt = (reference_tend - t_switch) / (reference_nlate - 1)
    p0["time"]["t_end_s"] = SENSITIVITY_T_END_S
    p0["time"]["late_n_points"] = int(round((SENSITIVITY_T_END_S - t_switch) / reference_late_dt)) + 1
    t_eval = f6.make_time_grid(p0)
    c_input = f6.common_source_input(p0)
    distances = [float(v) for v in p0["readout"]["distances_um"]]

    # Conventional reference is independent of D_relay,0.
    unorg = f6.simulate_unorganized(p0, t_eval, c_input)
    u20 = pd.DataFrame(f6.summarize(unorg, [TARGET_UM])).iloc[0]
    unorg_total_over_basal = (
        u20["peak_local_PKAc_release_uM_per_s"] / unorg["R_rest"]
    )

    rows = []
    for D0 in D_VALUES:
        p = copy.deepcopy(p0)
        p["RI"]["D_relay_0_um2_per_s"] = float(D0)
        org = f6.simulate_organized(p, t_eval, c_input)
        metrics = pd.DataFrame(f6.summarize(org, distances))
        lam, r2 = fit_attenuation(metrics)
        row20 = metrics.loc[
            (metrics["distance_um"] - TARGET_UM).abs().idxmin()
        ]

        peak_index = int(
            np.argmax(
                org["evoked"][
                    f6.nearest_index(org["x"], TARGET_UM)
                ]
            )
        )
        peak_at_window_edge = peak_index == (org["t"].size - 1)

        rows.append(
            {
                "D_relay_0_um2_per_s": D0,
                "D_relay_0_relative_to_reference": D0 / p0["RI"]["D_relay_0_um2_per_s"],
                "common_input_uM": c_input,
                "resting_PKAc_release_uM_per_s": org["R_rest"],
                "peak_cAMP_20um_uM": row20["peak_cAMP_uM"],
                "peak_total_PKAc_20um_uM_per_s": row20["peak_local_PKAc_release_uM_per_s"],
                "peak_total_PKAc_20um_over_basal": row20["peak_local_PKAc_release_uM_per_s"] / org["R_rest"],
                "peak_evoked_PKAc_20um_uM_per_s": row20["peak_evoked_PKAc_release_uM_per_s"],
                "peak_evoked_PKAc_20um_over_basal": row20["peak_evoked_PKAc_release_uM_per_s"] / org["R_rest"],
                "time_peak_evoked_20um_s": row20["time_peak_evoked_PKAc_release_s"],
                "peak_at_window_edge": peak_at_window_edge,
                "sensitivity_t_end_s": SENSITIVITY_T_END_S,
                "AUC_evoked_20um_uM": row20["AUC_evoked_PKAc_release_uM"],
                "lambda_app_um": lam,
                "lambda_fit_R2": r2,
                "conventional_20um_total_PKAc_over_basal": unorg_total_over_basal,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "Fig6_Drelay_sensitivity.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(9.4, 3.0))

    ax = axes[0]
    ax.plot(df["D_relay_0_um2_per_s"], df["peak_total_PKAc_20um_over_basal"], "o-")
    ax.axhline(1.0, ls="--", lw=1)
    ax.axhline(unorg_total_over_basal, ls=":", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel(r"$D_{relay,0}$ ($\mu$m$^2$ s$^{-1}$)")
    ax.set_ylabel("20-µm peak local PKAc / basal")
    ax.set_title("A")

    ax = axes[1]
    ax.plot(df["D_relay_0_um2_per_s"], 1000.0 * df["peak_cAMP_20um_uM"], "o-")
    ax.axhline(1000.0 * p0["cAMP"]["rest_uM"], ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel(r"$D_{relay,0}$ ($\mu$m$^2$ s$^{-1}$)")
    ax.set_ylabel("20-µm peak cAMP (nM)")
    ax.set_title("B")

    ax = axes[2]
    ax.plot(df["D_relay_0_um2_per_s"], df["lambda_app_um"], "o-")
    ax.set_xscale("log")
    ax.set_xlabel(r"$D_{relay,0}$ ($\mu$m$^2$ s$^{-1}$)")
    ax.set_ylabel(r"Apparent attenuation length $\lambda_{app}$ (µm)")
    ax.set_title("C")

    fig.tight_layout()
    fig.savefig(OUT / "Fig6_Drelay_sensitivity.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "Fig6_Drelay_sensitivity.pdf", bbox_inches="tight")
    plt.close(fig)

    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
