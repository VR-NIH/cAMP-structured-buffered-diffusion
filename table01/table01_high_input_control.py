#!/usr/bin/env python3
"""Generate Table 1: strong-input organized-versus-radial transport comparison.

The calculation uses the Figure 7 geometry, 1-s stimulus, cAMP clearance,
RI activation, and local non-depleting PKAc readout. The proximal free-cAMP
concentration during the pulse is set to 2.5 uM for all three branches:

1. conventional radial diffusion with rapid-equilibrium immobile buffering;
2. radial diffusion with the same spherical geometry but no buffer retardation,
   so cAMP diffuses at D_free; and
3. the organized-path high-recapture regime with occupancy-dependent
   structured mobility.

The radial no-buffer branch tests whether removal of conventional buffer
retardation and use of the full free-diffusion coefficient are sufficient to
reproduce the organized response. It is not a one-variable isolation of
geometry because the organized branch also has a different constitutive
mobility law and reduced productive coordinate.

Outputs:
  outputs/Table1_high_input_transport_comparison.csv
  outputs/Table1_high_input_diagnostics.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
FIG7_DIR = HERE.parent / "fig07"
if str(FIG7_DIR) not in sys.path:
    sys.path.insert(0, str(FIG7_DIR))

import fig07_simulation as f7

PARAMS = FIG7_DIR / "fig07_parameters.json"
OUT = HERE / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

PROXIMAL_INPUT_UM = 2.5
TARGET_UM = 20.0


def simulate_radial_no_buffer(p: dict, t_eval: np.ndarray, c_input: float) -> dict:
    """Radial 3D branch with D_free and no rapid-equilibrium buffer retardation."""
    S, _, f_ss = f7.common_functions(p)

    a = p["geometry"]["source_radius_um"]
    zc = p["geometry"]["common_coupling_distance_um"]
    L = p["geometry"]["domain_um"]
    N = int(p["geometry"]["n_nodes_unorganized"])
    x = np.linspace(zc, L, N)
    r = a + x
    dr = r[1] - r[0]

    c_rest = p["cAMP"]["rest_uM"]
    D_free = p["cAMP"]["D_free_um2_per_s"]
    k_clear = p["cAMP"]["k_clear_per_s"]

    kon = p["activation"]["kon_A_per_uM_s"]
    koff = p["activation"]["koff_A_per_s"]
    k_rel = p["activation"]["k_rel_per_s"]
    RT = p["RI"]["RT_uM"]
    HT = RT / p["RI"]["rho_R_to_C"]

    f0 = float(f_ss(c_rest))
    R_rest = k_rel * f0 * f0 * HT
    M = N - 1

    # Spherical finite-volume geometry, matching the conventional Fig. 7 branch.
    r_face = 0.5 * (r[:-1] + r[1:])
    inner = r_face
    outer = np.r_[r_face[1:], r[-1]]
    volume_factor = (outer**3 - inner**3) / 3.0

    def rhs(t, y, boundary_c):
        c_int = np.clip(y[:M], 1e-12, 1e5)
        f = np.clip(y[M:], 0.0, 1.0)
        c = np.r_[boundary_c, c_int]

        q_out = -D_free * np.diff(c) / dr
        flux = r_face * r_face * q_out

        dc = np.empty(M)
        dc[:-1] = (flux[:-1] - flux[1:]) / volume_factor[:-1]
        dc[-1] = flux[-1] / volume_factor[-1]
        dc -= k_clear * (c_int - c_rest)

        df = kon * c * S(c) * (1.0 - f) - koff * f
        return np.r_[dc, df]

    t, c, f = f7.integrate_fixed_boundary(
        p, t_eval, rhs, N, c_input, c_rest, f0
    )
    release = k_rel * f * f * HT
    evoked = release - R_rest
    return {
        "branch": "Radial no buffer",
        "x": x,
        "t": t,
        "c": c,
        "f": f,
        "release": release,
        "evoked": evoked,
        "R_rest": R_rest,
        "c_input": c_input,
    }


def source_equivalent(p: dict, c_input: float) -> dict:
    """Equivalent disk-source quantities under the Figure 7 calibration."""
    c_rest = p["cAMP"]["rest_uM"]
    D_free = p["cAMP"]["D_free_um2_per_s"]
    a = p["geometry"]["source_radius_um"]
    zc = p["geometry"]["common_coupling_distance_um"]
    conv = p["source_biology"]["conversion_molecules_per_uM_um3"]
    turnover = p["source_biology"]["turnover_cAMP_per_AC_per_s"]

    geometric_factor = np.sqrt(a * a + zc * zc) - zc
    J_source = (c_input - c_rest) * D_free / geometric_factor
    source_rate = J_source * conv * np.pi * a * a
    source_surface = c_rest + J_source * a / D_free
    return {
        "equivalent_source_rate_molecules_per_s": float(source_rate),
        "equivalent_active_AC_at_reference_turnover": float(source_rate / turnover),
        "source_surface_cAMP_uM": float(source_surface),
    }


def branch_summary(p: dict, b: dict, condition: str) -> tuple[dict, dict]:
    c_rest = p["cAMP"]["rest_uM"]
    D_free = p["cAMP"]["D_free_um2_per_s"]
    KB = p["RI"]["KB_uM"]
    KA = p["RI"]["KA_uM"]
    D0 = p["RI"]["D_relay_0_um2_per_s"]
    RT = p["RI"]["RT_uM"]
    KD = np.sqrt(KA * KB)

    i20 = f7.nearest_index(b["x"], TARGET_UM)
    c20 = b["c"][i20]
    release20 = b["release"][i20]
    j_c = int(np.argmax(c20))
    j_r = int(np.argmax(release20))

    mask = b["x"] <= TARGET_UM + 1e-9
    c_field = b["c"][mask]

    if condition == "organized_path":
        mobility_field = D0 * (c_field / (KB + c_field)) * (KA / (KA + c_field))
        mobility_at_input = D0 * (PROXIMAL_INPUT_UM / (KB + PROXIMAL_INPUT_UM)) * (KA / (KA + PROXIMAL_INPUT_UM))
    elif condition == "conventional_buffered_radial":
        kappa = RT * KD / (KD + c_field) ** 2
        mobility_field = D_free / (1.0 + kappa)
        kappa_input = RT * KD / (KD + PROXIMAL_INPUT_UM) ** 2
        mobility_at_input = D_free / (1.0 + kappa_input)
    elif condition == "radial_no_buffer":
        mobility_field = np.full_like(c_field, D_free)
        mobility_at_input = D_free
    else:
        raise ValueError(condition)

    peak_c_nM = 1000.0 * float(np.max(c20))
    evoked_c_nM = peak_c_nM - 1000.0 * c_rest
    peak_release_uM_s = float(np.max(release20))
    peak_over_basal = peak_release_uM_s / b["R_rest"]

    table_row = {
        "condition": condition,
        "max_local_mobility_scale_um2_per_s_0_20um": float(np.max(mobility_field)),
        "peak_cAMP_20um_nM": peak_c_nM,
        "evoked_peak_cAMP_20um_nM": evoked_c_nM,
        "peak_local_PKAc_over_basal": peak_over_basal,
        "peak_local_PKAc_release_nM_per_s": 1000.0 * peak_release_uM_s,
    }

    diag_row = {
        **table_row,
        "mobility_at_proximal_input_um2_per_s": float(mobility_at_input),
        "time_peak_cAMP_20um_s": float(b["t"][j_c]),
        "time_peak_PKAc_20um_s": float(b["t"][j_r]),
        "resting_PKAc_release_nM_per_s": 1000.0 * float(b["R_rest"]),
    }
    return table_row, diag_row


def main() -> None:
    with PARAMS.open("r", encoding="utf-8") as fh:
        p = json.load(fh)

    t_eval = f7.make_time_grid(p)
    c_input = PROXIMAL_INPUT_UM

    conventional = f7.simulate_unorganized(p, t_eval, c_input)
    no_buffer = simulate_radial_no_buffer(p, t_eval, c_input)
    organized = f7.simulate_organized(p, t_eval, c_input)

    branches = [
        (conventional, "conventional_buffered_radial"),
        (no_buffer, "radial_no_buffer"),
        (organized, "organized_path"),
    ]

    table_rows = []
    diag_rows = []
    for branch, condition in branches:
        table_row, diag_row = branch_summary(p, branch, condition)
        table_rows.append(table_row)
        diag_rows.append(diag_row)

    table = pd.DataFrame(table_rows)
    table.to_csv(OUT / "Table1_high_input_transport_comparison.csv", index=False)

    diagnostics = pd.DataFrame(diag_rows)
    src = source_equivalent(p, c_input)
    formal_max_uM_s = (
        p["activation"]["k_rel_per_s"]
        * p["RI"]["RT_uM"]
        / p["RI"]["rho_R_to_C"]
    )
    diagnostics["proximal_input_uM"] = c_input
    diagnostics["pulse_duration_s"] = p["time"]["stimulus_duration_s"]
    diagnostics["formal_max_PKAc_release_uM_per_s"] = formal_max_uM_s
    diagnostics["peak_release_percent_of_formal_max"] = (
        diagnostics["peak_local_PKAc_release_nM_per_s"] / (1000.0 * formal_max_uM_s) * 100.0
    )
    for key, value in src.items():
        diagnostics[key] = value
    diagnostics.to_csv(OUT / "Table1_high_input_diagnostics.csv", index=False)

    print("\nTable 1 values:\n")
    print(table.to_string(index=False))
    print("\nSource-equivalent diagnostics:\n")
    print(pd.Series(src).to_string())
    print(f"\nBasal local PKAc release = {diagnostics['resting_PKAc_release_nM_per_s'].iloc[0]:.6f} nM/s")
    print(f"Formal maximum local PKAc release = {formal_max_uM_s:.6f} uM/s")


if __name__ == "__main__":
    main()
