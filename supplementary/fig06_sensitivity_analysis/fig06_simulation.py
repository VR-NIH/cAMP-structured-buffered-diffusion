#!/usr/bin/env python3
"""Reproducible Fig. 6 organized-vs-unorganized RI capstone simulation.

This revision uses a *single biologically parameterized AC source* for both
transport formulations. A finite membrane-associated disk containing active
adenylyl cyclases defines the free-cAMP concentration sampled 20 nm from the source. That same proximal
waveform is imposed on:

  1) Unorganized RI: radially symmetric 3D conventional buffered diffusion.
  2) Organized RI: the reduced structured-buffered transport coordinate.

Thus AC production, source geometry, resting cAMP, RI abundance, clearance,
activation kinetics, and local non-depleting PKAc readout are identical. The
downstream transport formulation is the controlled difference.

Outputs written to the outputs/ subdirectory:
  Fig6_final_main.png/.pdf
  Fig6_diagnostic_traces_log.png/.pdf
  Fig6_metrics.csv
  Fig6_traces.csv
  Fig6_fields.npz

Run:
    python fig06_simulation.py fig06_parameters.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.stats import linregress
from scipy.sparse import lil_matrix
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
OUT.mkdir(parents=True, exist_ok=True)


def load_params(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def disk_axis_attenuation(a_um: float, z_um: float) -> float:
    """Normalized on-axis steady field of a uniformly emitting circular disk.

    eta(z) = [sqrt(a^2 + z^2) - z] / a, with eta(0)=1.
    """
    return (np.sqrt(a_um * a_um + z_um * z_um) - z_um) / a_um


def source_flux_density(p: dict) -> float:
    """Disk-source flux density derived from active AC production.

    The conversion 1 uM*um^3 = 602.214076 molecules converts the total
    molecular production rate to concentration-volume per unit time.
    """
    sb = p["source_biology"]
    a = p["geometry"]["source_radius_um"]
    q_mol_s = sb["active_AC_count"] * sb["turnover_cAMP_per_AC_per_s"]
    conv = sb["conversion_molecules_per_uM_um3"]
    return q_mol_s / (conv * np.pi * a * a)


def source_surface_concentration(p: dict) -> float:
    c0 = p["cAMP"]["rest_uM"]
    Dfree = p["cAMP"]["D_free_um2_per_s"]
    a = p["geometry"]["source_radius_um"]
    Jsrc = source_flux_density(p)
    return c0 + Jsrc * a / Dfree


def common_source_input(p: dict) -> float:
    """Common proximal free-cAMP concentration during the 1-s source event."""
    c0 = p["cAMP"]["rest_uM"]
    Jsrc = source_flux_density(p)
    Dfree = p["cAMP"]["D_free_um2_per_s"]
    a = p["geometry"]["source_radius_um"]
    zc = p["geometry"]["common_coupling_distance_um"]
    # Steady on-axis field of a uniformly emitting disk in a half-space:
    # c(z)-c0 = (Jsrc/Dfree) * [sqrt(a^2+z^2)-z].
    return c0 + (Jsrc / Dfree) * (np.sqrt(a*a + zc*zc) - zc)


def make_time_grid(p: dict) -> np.ndarray:
    t0 = p["time"]["stimulus_start_s"]
    dur = p["time"]["stimulus_duration_s"]
    t1 = t0 + dur
    dt = p["time"]["early_dt_s"]
    tend = p["time"]["t_end_s"]
    nlate = int(p["time"]["late_n_points"])
    early = np.arange(t0, t1 + 0.5 * dt, dt)
    late = np.linspace(t1, tend, nlate)
    return np.r_[early, late[1:]]


def common_functions(p: dict):
    KB = p["RI"]["KB_uM"]
    KA = p["RI"]["KA_uM"]
    kon = p["activation"]["kon_A_per_uM_s"]
    koff = p["activation"]["koff_A_per_s"]

    def S(c):
        return c / (KB + c)

    def A(c):
        return KA / (KA + c)

    def f_ss(c):
        rate_on = kon * c * S(c)
        return rate_on / (rate_on + koff)

    return S, A, f_ss


def fixed_boundary_jac_sparsity(N: int):
    """Jacobian sparsity for [N-1 cAMP unknowns, N activation fractions]."""
    M = N - 1
    nstate = M + N
    J = lil_matrix((nstate, nstate), dtype=int)
    for i in range(M):
        J[i, i] = 1
        if i > 0:
            J[i, i - 1] = 1
        if i < M - 1:
            J[i, i + 1] = 1
    for i in range(N):
        row = M + i
        J[row, row] = 1
        if i >= 1:
            J[row, i - 1] = 1
    return J.tocsr()


def integrate_fixed_boundary(p: dict, t_eval: np.ndarray, rhs_builder, N: int, c_input: float, c_rest: float, f0: float):
    """Integrate with common Dirichlet input during the pulse and rest afterward."""
    M = N - 1
    t_switch = p["time"]["stimulus_start_s"] + p["time"]["stimulus_duration_s"]
    early = t_eval[t_eval <= t_switch + 1e-12]
    late = t_eval[t_eval >= t_switch - 1e-12]
    method = p["numerics"]["method"]
    rtol = p["numerics"]["rtol"]
    atol = p["numerics"]["atol"]
    max_step = p["numerics"]["max_step_s"]
    sparsity = fixed_boundary_jac_sparsity(N)

    y0 = np.r_[np.full(M, c_rest), np.full(N, f0)]
    sol1 = solve_ivp(
        lambda t, y: rhs_builder(t, y, c_input),
        (early[0], early[-1]), y0, t_eval=early,
        method=method, rtol=rtol, atol=atol, max_step=max_step,
        jac_sparsity=sparsity,
    )
    if not sol1.success:
        raise RuntimeError(sol1.message)

    c1 = np.vstack([np.full(sol1.t.size, c_input), sol1.y[:M]])
    f1 = sol1.y[M:]
    y1 = np.r_[c1[1:, -1], f1[:, -1]]

    sol2 = solve_ivp(
        lambda t, y: rhs_builder(t, y, c_rest),
        (late[0], late[-1]), y1, t_eval=late,
        method=method, rtol=rtol, atol=atol, max_step=max_step,
        jac_sparsity=sparsity,
    )
    if not sol2.success:
        raise RuntimeError(sol2.message)

    c2 = np.vstack([np.full(sol2.t.size, c_rest), sol2.y[:M]])
    f2 = sol2.y[M:]

    t = np.r_[sol1.t, sol2.t[1:]]
    c = np.c_[c1, c2[:, 1:]]
    f = np.c_[f1, f2[:, 1:]]
    return t, c, f


def simulate_organized(p: dict, t_eval: np.ndarray, c_input: float):
    """Structured-buffered branch on the reduced RI coordinate s."""
    S, A, f_ss = common_functions(p)
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
        dc[-1] = -2.0 * q[-1] / ds  # reflecting distal boundary
        dc -= k_clear * (c_int - c_rest)

        df = kon * c * S(c) * (1.0 - f) - koff * f
        return np.r_[dc, df]

    t, c, f = integrate_fixed_boundary(p, t_eval, rhs, N, c_input, c_rest, f0)
    release = krel * f * f * HT
    evoked = release - R_rest

    return {
        "branch": "Organized RI",
        "x": s,
        "t": t,
        "c": c,
        "f": f,
        "release": release,
        "evoked": evoked,
        "R_rest": R_rest,
        "c_input": c_input,
    }


def simulate_unorganized(p: dict, t_eval: np.ndarray, c_input: float):
    """Conventional buffered-diffusion branch downstream of the common input.

    Downstream spreading is represented with radially symmetric 3D finite-volume
    transport. The proximal concentration waveform is identical to the organized
    branch, so source differences do not contribute to the comparison.
    """
    S, _, f_ss = common_functions(p)

    a = p["geometry"]["source_radius_um"]
    zc = p["geometry"]["common_coupling_distance_um"]
    L = p["geometry"]["domain_um"]
    N = int(p["geometry"]["n_nodes_unorganized"])
    x = np.linspace(zc, L, N)  # distance from source plane/surface
    r = a + x                   # radial coordinate used for 3D dilution
    dr = r[1] - r[0]

    c_rest = p["cAMP"]["rest_uM"]
    Dfree = p["cAMP"]["D_free_um2_per_s"]
    RT = p["RI"]["RT_uM"]
    KD = np.sqrt(p["RI"]["KA_uM"] * p["RI"]["KB_uM"])
    k_clear = p["cAMP"]["k_clear_per_s"]
    kon = p["activation"]["kon_A_per_uM_s"]
    koff = p["activation"]["koff_A_per_s"]
    krel = p["activation"]["k_rel_per_s"]
    HT = RT / p["RI"]["rho_R_to_C"]
    f0 = float(f_ss(c_rest))
    R_rest = krel * f0 * f0 * HT
    M = N - 1

    # Face radii and control volumes for unknown nodes 1..N-1. Node 0 is the
    # imposed common concentration boundary; the distal node is a half volume.
    rf = 0.5 * (r[:-1] + r[1:])
    inner = rf
    outer = np.r_[rf[1:], r[-1]]
    vol = (outer**3 - inner**3) / 3.0

    def kappa(c):
        return RT * KD / (KD + c) ** 2

    def rhs(t, y, boundary_c):
        c_int = np.clip(y[:M], 1e-12, 1e5)
        f = np.clip(y[M:], 0.0, 1.0)
        c = np.r_[boundary_c, c_int]

        # Rapid-equilibrium immobile-buffer reduction:
        # (1 + kappa(c)) dc/dt = Dfree * radial_laplacian(c)
        #                         - k_clear * (c - c_rest).
        # The buffer-capacity factor therefore divides the complete local
        # diffusion-plus-clearance balance; it is not placed inside the flux.
        q_out = -Dfree * np.diff(c) / dr
        F = rf * rf * q_out

        dc = np.empty(M)
        dc[:-1] = (F[:-1] - F[1:]) / vol[:-1]
        dc[-1] = F[-1] / vol[-1]  # reflecting distal boundary
        dc -= k_clear * (c_int - c_rest)
        dc /= (1.0 + kappa(c_int))

        df = kon * c * S(c) * (1.0 - f) - koff * f
        return np.r_[dc, df]

    t, c, f = integrate_fixed_boundary(p, t_eval, rhs, N, c_input, c_rest, f0)
    release = krel * f * f * HT
    evoked = release - R_rest

    return {
        "branch": "Unorganized RI",
        "x": x,
        "t": t,
        "c": c,
        "f": f,
        "release": release,
        "evoked": evoked,
        "R_rest": R_rest,
        "c_input": c_input,
    }


def nearest_index(x: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(x - target)))


def summarize(branch: dict, distances: list[float]) -> list[dict]:
    rows = []
    for d in distances:
        i = nearest_index(branch["x"], d)
        ev = branch["evoked"][i]
        j = int(np.argmax(ev))
        rows.append({
            "branch": branch["branch"],
            "distance_um": d,
            "actual_grid_distance_um": float(branch["x"][i]),
            "peak_cAMP_uM": float(np.max(branch["c"][i])),
            "peak_local_PKAc_release_uM_per_s": float(np.max(branch["release"][i])),
            "peak_evoked_PKAc_release_uM_per_s": float(ev[j]),
            "time_peak_evoked_PKAc_release_s": float(branch["t"][j]),
            "AUC_evoked_PKAc_release_uM": float(np.trapezoid(np.maximum(ev, 0.0), branch["t"])),
        })
    return rows


def export_traces(branches: list[dict], distances: list[float], out: Path):
    rows = []
    for b in branches:
        for d in distances:
            i = nearest_index(b["x"], d)
            for k, t in enumerate(b["t"]):
                rows.append({
                    "branch": b["branch"],
                    "distance_um": d,
                    "time_s": float(t),
                    "local_PKAc_release_uM_per_s": float(b["release"][i, k]),
                    "evoked_PKAc_release_uM_per_s": float(b["evoked"][i, k]),
                    "cAMP_uM": float(b["c"][i, k]),
                })
    pd.DataFrame(rows).to_csv(out, index=False)


def render_main(p: dict, unorg: dict, org: dict, metrics: pd.DataFrame, out: Path):
    distances = p["readout"]["distances_um"]
    display_tmax = p["time"].get("kymograph_display_tmax_s", p["time"]["figure_display_tmax_s"])
    xlim = 20.0

    mask_u_t = unorg["t"] <= display_tmax
    mask_o_t = org["t"] <= display_tmax
    mask_u_x = unorg["x"] <= xlim
    mask_o_x = org["x"] <= xlim
    vmax = max(
        np.max(np.maximum(unorg["evoked"][np.ix_(mask_u_x, mask_u_t)], 0.0)),
        np.max(np.maximum(org["evoked"][np.ix_(mask_o_x, mask_o_t)], 0.0)),
    )

    fig = plt.figure(figsize=(11.4, 8.2))
    gs = fig.add_gridspec(3, 6, height_ratios=[0.75, 2.25, 1.6], hspace=0.82, wspace=1.05)

    axA = fig.add_subplot(gs[0, :])
    axA.axis("off")
    axA.text(0.02, 0.72, "A", transform=axA.transAxes, fontweight="bold", fontsize=12)
    sb = p["source_biology"]
    qsrc = sb["active_AC_count"] * sb["turnover_cAMP_per_AC_per_s"]
    csurf = source_surface_concentration(p)
    cin = common_source_input(p)
    axA.text(0.35, 0.84, "Localized active AC patch", fontweight="bold", transform=axA.transAxes)
    axA.text(0.29, 0.62, f"Q$_{{AC}}$ = {qsrc:.0f} cAMP s$^{{-1}}$; source surface = {1000*csurf:.0f} nM", transform=axA.transAxes)
    axA.text(0.30, 0.43, f"common input at 20 nm = {1000*cin:.0f} nM", transform=axA.transAxes)
    axA.text(0.10, 0.25, "Unorganized RI", fontweight="bold", transform=axA.transAxes)
    axA.text(0.10, 0.04, "3D geometric dilution + conventional buffered diffusion", transform=axA.transAxes)
    axA.text(0.62, 0.25, "Organized RI", fontweight="bold", transform=axA.transAxes)
    axA.text(0.62, 0.04, "structured buffered diffusion + distal local PKA", transform=axA.transAxes)

    axB = fig.add_subplot(gs[1, 0:3])
    axC = fig.add_subplot(gs[1, 3:6])

    for ax, b, letter, title, mt, mx in [
        (axB, unorg, "B", "Unorganized RI", mask_u_t, mask_u_x),
        (axC, org, "C", "Organized RI", mask_o_t, mask_o_x),
    ]:
        Z = np.maximum(b["evoked"][np.ix_(mx, mt)], 0.0)
        im = ax.pcolormesh(b["t"][mt], b["x"][mx], Z, shading="auto", vmin=0.0, vmax=vmax, cmap="magma")
        ax.set(xlabel="Time (s)", ylabel="Distance from source (µm)", title=title,
               xlim=(0, display_tmax), ylim=(0, xlim))
        ax.text(-0.13, 1.05, letter, transform=ax.transAxes, fontweight="bold", fontsize=12)
        ax.plot([0, 1], [20.8, 20.8], lw=3, clip_on=False)
    cb = fig.colorbar(im, ax=[axB, axC], fraction=0.026, pad=0.025)
    cb.set_label("Evoked local PKAc release (µM s$^{-1}$)")

    axD = fig.add_subplot(gs[2, 0:3])
    axE = fig.add_subplot(gs[2, 3:6])

    for name, marker in [("Unorganized RI", "o"), ("Organized RI", "s")]:
        dd = metrics[metrics.branch == name]
        axD.plot(dd.distance_um, dd.peak_evoked_PKAc_release_uM_per_s, marker=marker, label=name)
    axD.set_yscale("log")
    axD.set(xlabel="Distance from source (µm)", ylabel="Peak evoked local PKAc release\n(µM s$^{-1}$)")
    axD.text(0.0, 1.06, "D", transform=axD.transAxes, fontweight="bold", fontsize=12)

    do = metrics[metrics.branch == "Organized RI"].sort_values("distance_um")
    slope, intercept, r_att, _, _ = linregress(do.distance_um.values, np.log(do.peak_evoked_PKAc_release_uM_per_s.values))
    lambda_app = -1.0 / slope
    xx = np.linspace(min(distances), max(distances), 150)
    yy = np.exp(intercept + slope * xx)
    axD.plot(xx, yy, ls="--", label=f"organized fit, λapp={lambda_app:.2f} µm")
    axD.legend(frameon=False, fontsize=8)

    x2 = do.distance_um.values ** 2
    tp = do.time_peak_evoked_PKAc_release_s.values
    sl2, in2, r_lat, _, _ = linregress(x2, tp)
    axE.plot(x2, tp, "o")
    xx2 = np.linspace(0, max(x2) * 1.03, 150)
    axE.plot(xx2, in2 + sl2 * xx2, ls="--")
    axE.set(xlabel="Distance² (µm²)", ylabel="Time to peak local PKAc release (s)")
    axE.text(0.0, 1.06, "E", transform=axE.transAxes, fontweight="bold", fontsize=12)
    axE.text(0.05, 0.9, f"R² = {r_lat*r_lat:.3f}", transform=axE.transAxes)

    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    return lambda_app, r_att*r_att, sl2, in2, r_lat*r_lat


def render_supplement(p: dict, unorg: dict, org: dict, out: Path):
    distances = p["readout"]["distances_um"]
    fig, axs = plt.subplots(2, 2, figsize=(8.0, 5.8), sharex=True, sharey=True)
    for ax, d in zip(axs.flat, distances):
        for b in (unorg, org):
            i = nearest_index(b["x"], d)
            ax.plot(b["t"], b["release"][i], label=b["branch"])
        ax.axhline(org["R_rest"], ls="--", lw=0.9, label="resting output" if d == distances[0] else None)
        ax.set_yscale("log")
        ax.set_title(f"{d:g} µm")
        ax.set_xlim(0, p["time"].get("supplement_display_tmax_s", p["time"]["figure_display_tmax_s"]))
    axs[1, 0].set_xlabel("Time (s)")
    axs[1, 1].set_xlabel("Time (s)")
    axs[0, 0].set_ylabel("Local PKAc release (µM s$^{-1}$)")
    axs[1, 0].set_ylabel("Local PKAc release (µM s$^{-1}$)")
    axs[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    param_path = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "fig06_parameters.json"
    p = load_params(param_path)
    outdir = OUT
    t_eval = make_time_grid(p)
    c_input = common_source_input(p)
    Jsrc = source_flux_density(p)
    c_surface = source_surface_concentration(p)
    sb = p["source_biology"]
    qsrc = sb["active_AC_count"] * sb["turnover_cAMP_per_AC_per_s"]

    pd.DataFrame([{
        "active_AC_count": sb["active_AC_count"],
        "turnover_cAMP_per_AC_per_s": sb["turnover_cAMP_per_AC_per_s"],
        "total_cAMP_production_molecules_per_s": qsrc,
        "source_radius_um": p["geometry"]["source_radius_um"],
        "source_flux_density_uM_um_per_s": Jsrc,
        "rest_cAMP_uM": p["cAMP"]["rest_uM"],
        "source_surface_cAMP_uM": c_surface,
        "coupling_distance_um": p["geometry"]["common_coupling_distance_um"],
        "common_input_cAMP_uM": c_input,
    }]).to_csv(outdir / "Fig6_source_summary.csv", index=False)

    print(f"AC source: {sb['active_AC_count']} x {sb['turnover_cAMP_per_AC_per_s']:.1f} cAMP/s = {qsrc:.0f} molecules/s", flush=True)
    print(f"Derived disk flux density: {Jsrc:.6f} µM·µm/s", flush=True)
    print(f"Source-surface cAMP: {c_surface:.6f} µM", flush=True)
    print(f"Common proximal cAMP input at {1000*p['geometry']['common_coupling_distance_um']:.0f} nm: {c_input:.6f} µM", flush=True)
    print("Running unorganized branch ...", flush=True)
    unorg = simulate_unorganized(p, t_eval, c_input)
    print("Running organized branch ...", flush=True)
    org = simulate_organized(p, t_eval, c_input)

    distances = [float(v) for v in p["readout"]["distances_um"]]
    metrics = pd.DataFrame(summarize(unorg, distances) + summarize(org, distances))
    metrics.to_csv(outdir / "Fig6_metrics.csv", index=False)

    export_traces([unorg, org], distances, outdir / "Fig6_traces.csv")
    np.savez_compressed(
        outdir / "Fig6_fields.npz",
        t_unorganized=unorg["t"], x_unorganized=unorg["x"], c_unorganized=unorg["c"], evoked_unorganized=unorg["evoked"],
        t_organized=org["t"], x_organized=org["x"], c_organized=org["c"], evoked_organized=org["evoked"],
        common_input_uM=np.array([c_input]),
        source_surface_uM=np.array([c_surface]),
        source_flux_density_uM_um_per_s=np.array([Jsrc]),
        total_source_molecules_per_s=np.array([qsrc]),
    )

    lam, r2att, sl2, in2, r2lat = render_main(
        p, unorg, org, metrics, outdir / "Fig6_final_main"
    )
    render_supplement(p, unorg, org, outdir / "Fig6_diagnostic_traces_log")

    print(metrics.to_string(index=False))
    print(f"\nOrganized apparent attenuation length: {lam:.3f} µm (log-linear R²={r2att:.4f})")
    print(f"Organized latency fit: t_peak = {in2:.6f} + {sl2:.6f} x²; R²={r2lat:.4f}")
    print(f"Resting local PKAc release: {org['R_rest']:.9g} µM/s")


if __name__ == "__main__":
    main()
