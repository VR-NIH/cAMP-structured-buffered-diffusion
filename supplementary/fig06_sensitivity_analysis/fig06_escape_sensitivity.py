#!/usr/bin/env python3
"""Fig. 6 intermediate-regime sensitivity to lateral escape from the productive path.

This is a deliberately phenomenological extension of the reference organized-path
model. A first-order loss term, -k_escape * (c - c_rest), removes evoked free cAMP
from the reduced productive-path coordinate into an unresolved bulk compartment.
It is not cAMP degradation and no bulk return flux is resolved. k_escape = 0 is the
reference high-recapture organized-path limit.

All reference Fig. 6 source, geometry, RI, activation, clearance, and numerical
parameters are retained. The observation window is 30 s to resolve delayed peaks.

Outputs:
  outputs/Fig6_escape_sensitivity.csv
  outputs/Fig6_escape_thresholds.csv
  outputs/Fig6_escape_sensitivity.png
  outputs/Fig6_escape_sensitivity.pdf
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
TARGET_UM = 20.0
SENSITIVITY_T_END_S = 30.0

# Values reported explicitly in the manuscript/appendix table.
REPORT_ESCAPE_VALUES = np.array([0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0])
# Additional values provide a smooth monotonic curve and threshold interpolation.
DENSE_POSITIVE_VALUES = np.logspace(-3, np.log10(3.0), 21)
ESCAPE_VALUES = np.unique(np.r_[REPORT_ESCAPE_VALUES, DENSE_POSITIVE_VALUES])
OUTPUT_THRESHOLDS = [1.1, 1.5, 2.0]


def fit_attenuation(metrics: pd.DataFrame) -> tuple[float, float]:
    d = metrics.sort_values("distance_um")
    y = d["peak_evoked_PKAc_release_uM_per_s"].to_numpy()
    x = d["distance_um"].to_numpy()
    mask = y > 0
    if mask.sum() < 2:
        return np.nan, np.nan
    slope, _, r, _, _ = linregress(x[mask], np.log(y[mask]))
    if slope >= 0:
        return np.nan, r * r
    return -1.0 / slope, r * r


def simulate_organized_escape(p: dict, t_eval: np.ndarray, c_input: float, k_escape: float):
    """Reference organized branch plus phenomenological lateral escape."""
    S, A, f_ss = f6.common_functions(p)
    zc = p["geometry"]["common_coupling_distance_um"]
    L = p["geometry"]["domain_um"]
    N = int(p["geometry"]["n_nodes_organized"])
    s = np.linspace(zc, L, N)
    ds = s[1] - s[0]

    c_rest = p["cAMP"]["rest_uM"]
    D0 = p["RI"]["D_relay_0_um2_per_s"]
    k_clear = p["cAMP"]["k_clear_per_s"]
    kon = p["activation"]["kon_A_per_uM_s"]
    koff = p["activation"]["koff_A_per_s"]
    krel = p["activation"]["k_rel_per_s"]
    RT = p["RI"]["RT_uM"]
    HT = RT / p["RI"]["rho_R_to_C"]
    f0 = float(f_ss(c_rest))
    R_rest = krel * f0 * f0 * HT
    M = N - 1

    def rhs(t, y, boundary_c):
        c_int = np.clip(y[:M], 1e-12, 1e4)
        f = np.clip(y[M:], 0.0, 1.0)
        c = np.r_[boundary_c, c_int]

        D = D0 * S(c) * A(c)
        Dface = 2.0 * D[:-1] * D[1:] / (D[:-1] + D[1:] + 1e-30)
        q = Dface * np.diff(c) / ds

        dc = np.zeros(M)
        dc[:-1] = np.diff(q) / ds
        dc[-1] = -2.0 * q[-1] / ds
        # Reference PDE-mediated clearance removes evoked cAMP chemically.
        dc -= k_clear * (c_int - c_rest)
        # Lateral escape removes evoked cAMP from the productive coordinate into
        # an unresolved bulk compartment; it is not an additional PDE reaction.
        dc -= k_escape * (c_int - c_rest)

        df = kon * c * S(c) * (1.0 - f) - koff * f
        return np.r_[dc, df]

    t, c, f = f6.integrate_fixed_boundary(p, t_eval, rhs, N, c_input, c_rest, f0)
    release = krel * f * f * HT
    evoked = release - R_rest
    return {
        "branch": f"Organized RI, k_escape={k_escape:g} s^-1",
        "x": s,
        "t": t,
        "c": c,
        "f": f,
        "release": release,
        "evoked": evoked,
        "R_rest": R_rest,
        "c_input": c_input,
    }


def interpolate_threshold(df: pd.DataFrame, threshold: float) -> float:
    """Largest k_escape giving output >= threshold, interpolated in log(k)."""
    d = df[df["k_escape_per_s"] > 0].sort_values("k_escape_per_s")
    x = d["k_escape_per_s"].to_numpy()
    y = d["peak_total_PKAc_20um_over_basal"].to_numpy()
    if y[0] < threshold:
        return np.nan
    below = np.where(y < threshold)[0]
    if below.size == 0:
        return np.inf
    j = int(below[0])
    if j == 0:
        return np.nan
    x1, x2 = x[j - 1], x[j]
    y1, y2 = y[j - 1], y[j]
    # Linear interpolation of output against log escape rate.
    lx1, lx2 = np.log(x1), np.log(x2)
    frac = (threshold - y1) / (y2 - y1)
    return float(np.exp(lx1 + frac * (lx2 - lx1)))


def main() -> None:
    with PARAMS.open("r", encoding="utf-8") as fh:
        p0 = json.load(fh)

    # Extend observation to 30 s while retaining the reference late sampling interval.
    reference_tend = float(p0["time"]["t_end_s"])
    reference_nlate = int(p0["time"]["late_n_points"])
    t_switch = float(p0["time"]["stimulus_start_s"] + p0["time"]["stimulus_duration_s"])
    reference_late_dt = (reference_tend - t_switch) / (reference_nlate - 1)
    p0["time"]["t_end_s"] = SENSITIVITY_T_END_S
    p0["time"]["late_n_points"] = int(round((SENSITIVITY_T_END_S - t_switch) / reference_late_dt)) + 1

    t_eval = f6.make_time_grid(p0)
    c_input = f6.common_source_input(p0)
    distances = [float(v) for v in p0["readout"]["distances_um"]]

    # Matched conventional reference is unchanged by the organized-path escape parameter.
    unorg = f6.simulate_unorganized(p0, t_eval, c_input)
    u20 = pd.DataFrame(f6.summarize(unorg, [TARGET_UM])).iloc[0]
    unorg_total_over_basal = u20["peak_local_PKAc_release_uM_per_s"] / unorg["R_rest"]

    rows = []
    for k_escape in ESCAPE_VALUES:
        org = simulate_organized_escape(copy.deepcopy(p0), t_eval, c_input, float(k_escape))
        metrics = pd.DataFrame(f6.summarize(org, distances))
        lam, r2 = fit_attenuation(metrics)
        row20 = metrics.loc[(metrics["distance_um"] - TARGET_UM).abs().idxmin()]
        i20 = f6.nearest_index(org["x"], TARGET_UM)
        j = int(np.argmax(org["evoked"][i20]))

        rows.append({
            "k_escape_per_s": float(k_escape),
            "mean_productive_path_residence_s": np.inf if k_escape == 0 else 1.0 / float(k_escape),
            "common_input_uM": c_input,
            "D_relay_0_um2_per_s": p0["RI"]["D_relay_0_um2_per_s"],
            "k_clear_per_s": p0["cAMP"]["k_clear_per_s"],
            "resting_PKAc_release_uM_per_s": org["R_rest"],
            "peak_cAMP_20um_uM": row20["peak_cAMP_uM"],
            "peak_total_PKAc_20um_uM_per_s": row20["peak_local_PKAc_release_uM_per_s"],
            "peak_total_PKAc_20um_over_basal": row20["peak_local_PKAc_release_uM_per_s"] / org["R_rest"],
            "peak_evoked_PKAc_20um_uM_per_s": row20["peak_evoked_PKAc_release_uM_per_s"],
            "peak_evoked_PKAc_20um_over_basal": row20["peak_evoked_PKAc_release_uM_per_s"] / org["R_rest"],
            "time_peak_evoked_20um_s": row20["time_peak_evoked_PKAc_release_s"],
            "peak_at_window_edge": j == org["t"].size - 1,
            "sensitivity_t_end_s": SENSITIVITY_T_END_S,
            "AUC_evoked_20um_uM": row20["AUC_evoked_PKAc_release_uM"],
            "lambda_app_um": lam,
            "lambda_fit_R2": r2,
            "conventional_20um_total_PKAc_over_basal": unorg_total_over_basal,
        })

    df = pd.DataFrame(rows).sort_values("k_escape_per_s")
    df.to_csv(OUT / "Fig6_escape_sensitivity.csv", index=False)

    thresholds = []
    for thr in OUTPUT_THRESHOLDS:
        thresholds.append({
            "20um_total_PKAc_threshold_over_basal": thr,
            "approx_max_k_escape_per_s": interpolate_threshold(df, thr),
        })
    tdf = pd.DataFrame(thresholds)
    tdf.to_csv(OUT / "Fig6_escape_thresholds.csv", index=False)

    # Compact publication-style diagnostic figure.
    fig, axes = plt.subplots(1, 3, figsize=(9.4, 3.0))
    positive = df["k_escape_per_s"] > 0

    ax = axes[0]
    ax.plot(df.loc[positive, "k_escape_per_s"], df.loc[positive, "peak_total_PKAc_20um_over_basal"], "o-")
    ax.scatter([1e-3], [df.loc[df["k_escape_per_s"] == 0, "peak_total_PKAc_20um_over_basal"].iloc[0]], marker="s")
    for thr in OUTPUT_THRESHOLDS:
        ax.axhline(thr, ls="--", lw=0.8)
    ax.axhline(unorg_total_over_basal, ls=":", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel(r"Lateral escape rate $k_{escape}$ (s$^{-1}$)")
    ax.set_ylabel("20-µm peak local PKAc / basal")
    ax.set_title("A")

    ax = axes[1]
    ax.plot(df.loc[positive, "k_escape_per_s"], 1000.0 * df.loc[positive, "peak_cAMP_20um_uM"], "o-")
    ax.scatter([1e-3], [1000.0 * df.loc[df["k_escape_per_s"] == 0, "peak_cAMP_20um_uM"].iloc[0]], marker="s")
    ax.axhline(1000.0 * p0["cAMP"]["rest_uM"], ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel(r"Lateral escape rate $k_{escape}$ (s$^{-1}$)")
    ax.set_ylabel("20-µm peak cAMP (nM)")
    ax.set_title("B")

    ax = axes[2]
    ax.plot(df.loc[positive, "k_escape_per_s"], df.loc[positive, "lambda_app_um"], "o-")
    ax.scatter([1e-3], [df.loc[df["k_escape_per_s"] == 0, "lambda_app_um"].iloc[0]], marker="s")
    ax.set_xscale("log")
    ax.set_xlabel(r"Lateral escape rate $k_{escape}$ (s$^{-1}$)")
    ax.set_ylabel(r"Apparent attenuation length $\lambda_{app}$ (µm)")
    ax.set_title("C")

    fig.tight_layout()
    fig.savefig(OUT / "Fig6_escape_sensitivity.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "Fig6_escape_sensitivity.pdf", bbox_inches="tight")
    plt.close(fig)

    report = df[np.isclose(df["k_escape_per_s"].to_numpy()[:, None], REPORT_ESCAPE_VALUES[None, :], rtol=0, atol=1e-12).any(axis=1)]
    print("\nReported conditions:\n")
    print(report.to_string(index=False))
    print("\nInterpolated output thresholds:\n")
    print(tdf.to_string(index=False))


if __name__ == "__main__":
    main()
