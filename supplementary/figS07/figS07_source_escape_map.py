#!/usr/bin/env python3
"""Figure S7: source-amplitude x lateral-escape sensitivity for the Fig. 7 organized path.

Every grid point is a full organized-path simulation using the reference Fig. 7
spatial grid, 15-s observation window, activation kinetics, resting cAMP,
structured mobility, and fixed PDE-mediated clearance. The only swept variables are:
  1) the imposed proximal free-cAMP concentration during the 1-s source event; and
  2) the phenomenological lateral escape rate k_escape.

Lateral escape acts only on the evoked component:
    -k_escape * (c - c_rest)
and represents loss from the reduced productive-path coordinate into an unresolved
bulk compartment. Escaped cAMP is not returned to the productive coordinate.

Outputs:
  outputs/FigS7_source_escape_grid.csv
  outputs/FigS7_source_escape_representative.csv
  outputs/FigS7_source_escape_map.npz
  outputs/FigS7_source_escape_map_final.png
  outputs/FigS7_source_escape_map_final.pdf
  outputs/FigS7_source_escape_monotonicity.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

HERE = Path(__file__).resolve().parent
FIG7_DIR = HERE.parent / "fig07"
if str(FIG7_DIR) not in sys.path:
    sys.path.insert(0, str(FIG7_DIR))

import fig07_simulation as f7
from fig07_escape_sensitivity import simulate_organized_escape

OUT = HERE / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
PARAMS = FIG7_DIR / "fig07_parameters.json"
TARGET_UM = 20.0
N_JOBS = 1


def unique_sorted(values, ndigits=12):
    vals = sorted({round(float(v), ndigits) for v in values})
    return np.array(vals, dtype=float)


def grid_centers_to_edges(v):
    """Edges for monotonically increasing centers in a transformed coordinate."""
    v = np.asarray(v, dtype=float)
    mids = 0.5 * (v[:-1] + v[1:])
    e0 = v[0] - (mids[0] - v[0])
    e1 = v[-1] + (v[-1] - mids[-1])
    return np.r_[e0, mids, e1]


def escape_plot_coordinate(k, kmin_positive):
    """Log10 coordinate with a dedicated reference column for k=0."""
    if k == 0:
        return np.log10(kmin_positive) - 0.34
    return np.log10(k)


def run_point(p, t_eval, input_uM, k_escape):
    b = simulate_organized_escape(p, t_eval, float(input_uM), float(k_escape))
    i = f7.nearest_index(b["x"], TARGET_UM)
    ev = b["evoked"][i]
    j = int(np.argmax(ev))
    peak_c = float(np.max(b["c"][i]))
    peak_release = float(np.max(b["release"][i]))
    return {
        "proximal_input_uM": float(input_uM),
        "proximal_input_nM": 1000.0 * float(input_uM),
        "k_escape_per_s": float(k_escape),
        "peak_cAMP_20um_uM": peak_c,
        "delta_peak_cAMP_20um_nM": 1000.0 * (peak_c - p["cAMP"]["rest_uM"]),
        "peak_total_PKAc_20um_uM_per_s": peak_release,
        "peak_total_PKAc_20um_over_basal": peak_release / b["R_rest"],
        "peak_evoked_PKAc_20um_uM_per_s": float(np.max(ev)),
        "time_peak_evoked_20um_s": float(b["t"][j]),
        "peak_at_window_edge": bool(j == len(b["t"]) - 1),
    }


def run_input_row(args):
    """Run all escape values for one proximal input, then let worker exit."""
    p, t_eval, input_uM, escape_values = args
    return [run_point(p, t_eval, input_uM, k) for k in escape_values]


def monotonic_nonincreasing(y, tol=1e-8):
    y = np.asarray(y, dtype=float)
    return bool(np.all(np.diff(y) <= tol))


def main():
    with PARAMS.open("r", encoding="utf-8") as fh:
        p = json.load(fh)

    # Keep the reference Fig. 7 15-s window. Peaks across this sweep are checked
    # explicitly below to ensure none occurs at the observation boundary.
    t_eval = f7.make_time_grid(p)
    ref_input = float(f7.common_source_input(p))

    # Proximal total free-cAMP concentration during the 1-s source event.
    # A logarithmic base grid is augmented with the exact Fig. 7 reference input
    # and the 500- and 1000-nM representative conditions.
    base_inputs = np.logspace(np.log10(0.05), np.log10(1.0), 8)
    inputs = unique_sorted(np.r_[base_inputs, ref_input, 0.5, 1.0])

    # Positive escape values span 1e-3 to 3 s^-1 logarithmically and are augmented
    # by values used in the text/table. Zero is retained as a separate reference.
    base_escape = np.array([0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0])
    escape = unique_sorted(np.r_[0.0, base_escape, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0])

    n_tasks = len(inputs) * len(escape)
    print(f"Running {n_tasks} full organized-path simulations "
          f"({len(inputs)} inputs x {len(escape)} escape values) serially...")
    checkpoint = OUT / "FigS7_source_escape_grid_partial.csv"
    rows = []
    completed_inputs = set()
    if checkpoint.exists():
        prior = pd.read_csv(checkpoint)
        rows = prior.to_dict("records")
        counts = prior.groupby("proximal_input_uM")["k_escape_per_s"].nunique()
        completed_inputs = {float(ci) for ci, n in counts.items() if int(n) == len(escape)}
        print(f"Resuming from checkpoint with {len(completed_inputs)} complete input rows.", flush=True)
    for ii, ci in enumerate(inputs, start=1):
        if any(np.isclose(ci, done, rtol=0, atol=1e-10) for done in completed_inputs):
            print(f"  input row {ii}/{len(inputs)} already complete: {1000*ci:.3f} nM", flush=True)
            continue
        print(f"  input row {ii}/{len(inputs)}: {1000*ci:.3f} nM", flush=True)
        rows.extend(run_input_row((p, t_eval, float(ci), escape)))
        pd.DataFrame(rows).to_csv(checkpoint, index=False)
    df = pd.DataFrame(rows).sort_values(["proximal_input_uM", "k_escape_per_s"]).reset_index(drop=True)
    if checkpoint.exists():
        checkpoint.unlink()
    df["is_reference_input"] = np.isclose(df["proximal_input_uM"], ref_input, rtol=0, atol=1e-12)
    df["is_reference_condition"] = df["is_reference_input"] & np.isclose(df["k_escape_per_s"], 0.0)
    df.to_csv(OUT / "FigS7_source_escape_grid.csv", index=False)

    # Matrices ordered exactly as the axes.
    pk = df.pivot(index="proximal_input_uM", columns="k_escape_per_s", values="peak_total_PKAc_20um_over_basal").reindex(index=inputs, columns=escape).to_numpy()
    dc = df.pivot(index="proximal_input_uM", columns="k_escape_per_s", values="delta_peak_cAMP_20um_nM").reindex(index=inputs, columns=escape).to_numpy()
    tp = df.pivot(index="proximal_input_uM", columns="k_escape_per_s", values="time_peak_evoked_20um_s").reindex(index=inputs, columns=escape).to_numpy()

    # Save the numerical grid in compact form as well as long-form CSV.
    np.savez_compressed(
        OUT / "FigS7_source_escape_map.npz",
        proximal_input_uM=inputs,
        k_escape_per_s=escape,
        peak_PKAc_over_basal=pk,
        delta_peak_cAMP_nM=dc,
        time_peak_evoked_s=tp,
        reference_input_uM=ref_input,
    )

    # Representative panel-C curves use exact 242-ish reference, 500, and 1000 nM inputs.
    reps = []
    for value, label in [(ref_input, "reference (~242 nM)"), (0.5, "500 nM"), (1.0, "1000 nM")]:
        d = df[np.isclose(df["proximal_input_uM"], value, rtol=0, atol=1e-12)].copy()
        d["representative_input"] = label
        reps.append(d)
    rep = pd.concat(reps, ignore_index=True)
    rep.to_csv(OUT / "FigS7_source_escape_representative.csv", index=False)

    # Diagnostics: whether added escape ever raises the 20-um cAMP peak at fixed input.
    mono_rows = []
    for ci in inputs:
        d = df[np.isclose(df["proximal_input_uM"], ci, rtol=0, atol=1e-12)].sort_values("k_escape_per_s")
        cvals = d["peak_cAMP_20um_uM"].to_numpy()
        pvals = d["peak_total_PKAc_20um_over_basal"].to_numpy()
        mono_rows.append({
            "proximal_input_nM": 1000.0 * ci,
            "cAMP_peak_nonincreasing_with_escape": monotonic_nonincreasing(cvals, tol=1e-10),
            "PKAc_peak_nonincreasing_with_escape": monotonic_nonincreasing(pvals, tol=1e-10),
            "max_positive_cAMP_step_nM": 1000.0 * max(0.0, float(np.max(np.diff(cvals)))),
        })
    mono = pd.DataFrame(mono_rows)
    mono.to_csv(OUT / "FigS7_source_escape_monotonicity.csv", index=False)

    if df["peak_at_window_edge"].any():
        edge = df[df["peak_at_window_edge"]]
        raise RuntimeError(f"Some response peaks occur at the 15-s window edge:\n{edge}")

    # ---------- publication-style Figure S7 ----------
    xcoord = np.array([escape_plot_coordinate(k, escape[1]) for k in escape])
    ycoord = np.log10(inputs * 1000.0)  # nM, plotted logarithmically
    xedge = grid_centers_to_edges(xcoord)
    yedge = grid_centers_to_edges(ycoord)

    fig = plt.figure(figsize=(8.2, 7.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.9], hspace=0.38, wspace=0.31)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, :])

    # Panel A: local PKAc / basal. Log normalization resolves the near-basal region.
    vmax_pk = max(40.0, float(np.nanmax(pk)))
    imA = axA.pcolormesh(xedge, yedge, pk, shading="flat", norm=LogNorm(vmin=1.0, vmax=vmax_pk))
    levels_pk = [1.1, 1.5, 2.0, 5.0, 10.0]
    csA = axA.contour(xcoord, ycoord, pk, levels=levels_pk, linewidths=0.8)
    axA.clabel(csA, inline=True, fontsize=7, fmt=lambda v: f"{v:g}x")
    cbA = fig.colorbar(imA, ax=axA, fraction=0.05, pad=0.035)
    cbA.set_label("20-µm peak local PKAc / basal")

    # Panel B: stimulus-evoked peak free cAMP above the 25-nM resting background.
    imB = axB.pcolormesh(xedge, yedge, dc, shading="flat")
    dcmax = float(np.nanmax(dc))
    levels_dc = [v for v in [1, 2, 5, 10, 20, 40] if v < dcmax]
    if levels_dc:
        csB = axB.contour(xcoord, ycoord, dc, levels=levels_dc, linewidths=0.8)
        axB.clabel(csB, inline=True, fontsize=7, fmt=lambda v: f"{v:g}")
    cbB = fig.colorbar(imB, ax=axB, fraction=0.05, pad=0.035)
    cbB.set_label(r"20-µm $\Delta c_{peak}$ (nM)")

    # Shared heatmap formatting and reference point.
    x_tick_vals = [0.0, 1e-3, 1e-2, 1e-1, 1.0, 3.0]
    x_tick_pos = [escape_plot_coordinate(v, escape[1]) for v in x_tick_vals]
    x_tick_labels = ["0", r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$", "1", "3"]
    y_tick_vals = [50, 100, 200, 500, 1000]
    y_tick_pos = np.log10(y_tick_vals)
    for ax, letter, title in [
        (axA, "A", "Distal local PKA output"),
        (axB, "B", "Distal stimulus-evoked free cAMP"),
    ]:
        ax.set_xticks(x_tick_pos, x_tick_labels)
        ax.set_yticks(y_tick_pos, [str(v) for v in y_tick_vals])
        ax.set_xlabel(r"Lateral escape rate $k_{escape}$ (s$^{-1}$)")
        ax.set_ylabel("Proximal cAMP input (nM)")
        ax.set_title(title, fontsize=10)
        ax.text(-0.17, 1.05, letter, transform=ax.transAxes, fontweight="bold", fontsize=12)
        xref = escape_plot_coordinate(0.0, escape[1])
        yref = np.log10(ref_input * 1000.0)
        ax.scatter([xref], [yref], marker="s", s=30, zorder=5)
        # Visually separate the dedicated zero-escape reference column from log-positive values.
        ax.axvline(0.5 * (xcoord[0] + xcoord[1]), lw=0.7, ls="--")

    # Panel C: representative curves. Use same transformed x coordinate as A/B.
    for value, label, marker in [
        (ref_input, "Reference input (~242 nM)", "o"),
        (0.5, "500 nM", "s"),
        (1.0, "1000 nM", "^"),
    ]:
        d = df[np.isclose(df["proximal_input_uM"], value, rtol=0, atol=1e-12)].sort_values("k_escape_per_s")
        xc = np.array([escape_plot_coordinate(v, escape[1]) for v in d["k_escape_per_s"]])
        axC.plot(xc, d["peak_total_PKAc_20um_over_basal"], marker=marker, ms=3.2, lw=1.2, label=label)
    axC.axhline(1.0, lw=0.8, ls="--")
    axC.set_yscale("log")
    axC.set_xticks(x_tick_pos, x_tick_labels)
    axC.set_xlabel(r"Lateral escape rate $k_{escape}$ (s$^{-1}$)")
    axC.set_ylabel("20-µm peak local PKAc / basal")
    axC.set_title("Representative source-amplitude dependence", fontsize=10)
    axC.text(-0.08, 1.04, "C", transform=axC.transAxes, fontweight="bold", fontsize=12)
    axC.legend(frameon=False, ncol=3, loc="upper right")
    axC.set_ylim(0.95, vmax_pk * 1.08)

    fig.suptitle("Figure S7. Source amplitude and lateral escape jointly determine distal organized-path signaling", fontsize=11, y=0.995)
    fig.savefig(OUT / "FigS7_source_escape_map_final.png", dpi=400, bbox_inches="tight")
    fig.savefig(OUT / "FigS7_source_escape_map_final.pdf", bbox_inches="tight")
    plt.close(fig)

    ref = df[df["is_reference_condition"]].iloc[0]
    print("\nReference condition:")
    print(ref[["proximal_input_nM", "k_escape_per_s", "peak_cAMP_20um_uM", "delta_peak_cAMP_20um_nM", "peak_total_PKAc_20um_over_basal", "time_peak_evoked_20um_s"]].to_string())
    print("\nRepresentative anchor values:")
    anchors = rep[rep["k_escape_per_s"].isin([0.0, 0.1, 0.3, 1.0, 3.0])]
    print(anchors[["representative_input", "k_escape_per_s", "peak_cAMP_20um_uM", "delta_peak_cAMP_20um_nM", "peak_total_PKAc_20um_over_basal", "time_peak_evoked_20um_s"]].to_string(index=False))
    print("\nMonotonicity check:")
    print(mono.to_string(index=False))
    print("\nGrid min/max:")
    print(f"PKAc/basal: {np.nanmin(pk):.6g} to {np.nanmax(pk):.6g}")
    print(f"delta cAMP peak: {np.nanmin(dc):.6g} to {np.nanmax(dc):.6g} nM")
    print(f"Reference input = {1000*ref_input:.6f} nM")


if __name__ == "__main__":
    main()
