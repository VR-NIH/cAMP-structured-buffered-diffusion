"""
Revised Figure 2: occupancy-dependent RI mobility, PDE range control,
and equal-input temporal packaging.

Key revisions relative to the previous script:
  1. Correct half-cell boundary treatment for flux-driven panels E-F.
  2. Use a reflecting proximal boundary after the 1-s concentration clamp
     in panels A-D, eliminating post-stimulus drainage through x=0.
  3. Apply half-cell corrections at reflecting boundaries.
  4. Label distance as an effective RI-network coordinate (s), not as
     physical one-dimensional Brownian diffusion.
  5. Label k_PDE as effective PDE-mediated clearance.
  6. Remove the empirical power-law fit from panel C; the line joining
     simulation points is a visual guide only.
  7. Report panel-F integral as E_relay = integral D_relay dt, with units um^2.

The script reads all numerical parameters from Fig2_parameters.json in the
same directory unless --params is supplied.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker


def parse_args():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--params",
        type=Path,
        default=here / "fig02_parameters.json",
        help="Path to JSON parameter file.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=here / "outputs",
        help="Directory for figure and metrics outputs.",
    )
    return parser.parse_args()


def load_parameters(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main():
    args = parse_args()
    P = load_parameters(args.params)
    args.outdir.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # Styling
    # -----------------------------
    plt.rcParams.update({
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
        "figure.titlesize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    # ============================================================
    # Shared mobility parameters
    # ============================================================
    KB = float(P["mobility"]["K_B_uM"])
    KA = float(P["mobility"]["K_A_uM"])
    D0 = float(P["mobility"]["D0_um2_per_s"])
    CMAX = np.sqrt(KB * KA)

    def deff(c):
        v = np.clip(c, 1e-12, 1e4)
        return D0 * (v / (KB + v)) * (KA / (KA + v))

    solver = P["solver"]
    method = solver["method"]
    rtol = float(solver["rtol"])
    atol = float(solver["atol"])
    max_step = float(solver["max_step_s"])

    # ============================================================
    # Panels A-D: fixed proximal concentration pulse
    # ============================================================
    A = P["panels_A_D"]
    SRC_CONC = float(A["source_concentration_uM"])
    SRC_OFF = float(A["source_duration_s"])
    K_BASE = float(A["baseline_kPDE_per_s"])
    L = float(A["domain_length_um"])
    N = int(A["grid_points"])
    T_MAX = float(A["simulation_end_s"])
    N_EVAL = int(A["evaluation_points"])
    X_MAX_PLOT = float(A["plot_distance_max_um"])
    T_MAX_PLOT = float(A["plot_time_max_s"])
    PEN_FRAC = float(A["penetration_threshold_fraction"])

    s = np.linspace(0.0, L, N)
    ds = s[1] - s[0]

    def make_model(kPDE, proximal_boundary):
        if proximal_boundary not in {"clamped", "reflecting"}:
            raise ValueError("proximal_boundary must be 'clamped' or 'reflecting'")

        def model(t, C):
            C = np.clip(C, 0.0, 1e4)
            dCdt = np.zeros_like(C)

            # Arithmetic interface averaging retained from the original A-D model.
            D_mid = 0.5 * (deff(C[:-1]) + deff(C[1:]))
            flux = D_mid * (C[1:] - C[:-1]) / ds

            dCdt[1:-1] = (flux[1:] - flux[:-1]) / ds

            # Boundary nodes are half-width control volumes. The distal boundary
            # is reflecting throughout. After source removal, the proximal
            # boundary is also reflecting.
            if proximal_boundary == "reflecting":
                dCdt[0] = 2.0 * flux[0] / ds
            else:
                # Proximal Dirichlet value is imposed by the initial value and
                # zero time derivative during the clamp interval.
                dCdt[0] = 0.0
            dCdt[-1] = -2.0 * flux[-1] / ds

            dCdt -= kPDE * C

            # The clamped boundary must remain fixed despite the PDE sink.
            if proximal_boundary == "clamped":
                dCdt[0] = 0.0
            return dCdt

        return model

    def run_sim(kPDE):
        model_clamped = make_model(kPDE, "clamped")
        model_reflecting = make_model(kPDE, "reflecting")
        t_eval = np.linspace(0.0, T_MAX, N_EVAL)

        C0 = np.zeros(N)
        C0[0] = SRC_CONC

        # During the pulse, the proximal concentration is clamped to SRC_CONC.
        t_ev1 = t_eval[t_eval <= SRC_OFF]
        if len(t_ev1) == 0 or t_ev1[-1] < SRC_OFF:
            t_ev1 = np.append(t_ev1, SRC_OFF)
        sol1 = solve_ivp(
            model_clamped,
            [0.0, SRC_OFF],
            C0,
            method=method,
            t_eval=t_ev1,
            rtol=rtol,
            atol=atol,
            max_step=max_step,
        )
        if not sol1.success:
            raise RuntimeError(sol1.message)
        sol1.y[0, :] = SRC_CONC

        # Source removal: release the concentration clamp and impose zero
        # proximal flux. Existing cAMP at the boundary is retained and can
        # redistribute inward or be removed by PDE; it cannot drain out through s=0.
        C1 = np.clip(sol1.y[:, -1], 0.0, 1e4)
        t_ev2 = t_eval[t_eval >= SRC_OFF]
        if len(t_ev2) == 0 or t_ev2[0] > SRC_OFF:
            t_ev2 = np.insert(t_ev2, 0, SRC_OFF)
        sol2 = solve_ivp(
            model_reflecting,
            [SRC_OFF, T_MAX],
            C1,
            method=method,
            t_eval=t_ev2,
            rtol=rtol,
            atol=atol,
            max_step=max_step,
        )
        if not sol2.success:
            raise RuntimeError(sol2.message)

        # Avoid duplicating the switch time when concatenating the two phases.
        start2 = 1 if (len(sol2.t) and np.isclose(sol2.t[0], sol1.t[-1])) else 0
        y_full = np.hstack([sol1.y, sol2.y[:, start2:]])
        t_full = np.hstack([sol1.t, sol2.t[start2:]])
        mx = np.max(y_full, axis=1)
        return y_full, t_full, mx

    def penetration_depth(mx):
        """10%-of-peak penetration depth with linear interpolation.

        Returns NaN if the threshold is not reached within the domain.
        """
        peak = float(np.max(mx))
        threshold = PEN_FRAC * peak
        below = np.where(mx < threshold)[0]
        if len(below) == 0:
            return np.nan
        i = int(below[0])
        if i == 0:
            return float(s[0])
        x0, x1 = float(s[i - 1]), float(s[i])
        y0, y1 = float(mx[i - 1]), float(mx[i])
        if np.isclose(y1, y0):
            return x1
        frac = (threshold - y0) / (y1 - y0)
        return x0 + frac * (x1 - x0)

    print("Running Fig. 2 A-D baseline simulation...")
    y_base, t_base, mx_base = run_sim(K_BASE)
    baseline_depth = penetration_depth(mx_base)

    print("Sweeping effective PDE-mediated clearance...")
    k_vals = np.logspace(
        np.log10(float(A["PDE_sweep_min_per_s"])),
        np.log10(float(A["PDE_sweep_max_per_s"])),
        int(A["PDE_sweep_points"]),
    )
    depths = []
    for k in k_vals:
        _, _, mx = run_sim(k)
        depths.append(penetration_depth(mx))
    depths = np.asarray(depths, dtype=float)

    # Local log-log clearance slope for the structured-mobility model.
    valid = np.isfinite(depths) & (depths > 0)
    alpha_struct = np.full_like(depths, np.nan)
    alpha_struct[valid] = np.gradient(np.log(depths[valid]), np.log(k_vals[valid]))

    # Protocol-matched constant-mobility control. The constant mobility is
    # set to the maximum structured mobility D_relay(c_max).
    D_CONST = float(deff(CMAX))

    def make_constant_model(kPDE, proximal_boundary):
        if proximal_boundary not in {"clamped", "reflecting"}:
            raise ValueError("proximal_boundary must be 'clamped' or 'reflecting'")

        def model(t, C):
            C = np.clip(C, 0.0, 1e4)
            dCdt = np.zeros_like(C)
            flux = D_CONST * (C[1:] - C[:-1]) / ds
            dCdt[1:-1] = (flux[1:] - flux[:-1]) / ds
            if proximal_boundary == "reflecting":
                dCdt[0] = 2.0 * flux[0] / ds
            else:
                dCdt[0] = 0.0
            dCdt[-1] = -2.0 * flux[-1] / ds
            dCdt -= kPDE * C
            if proximal_boundary == "clamped":
                dCdt[0] = 0.0
            return dCdt

        return model

    def run_constant_sim(kPDE):
        model_clamped = make_constant_model(kPDE, "clamped")
        model_reflecting = make_constant_model(kPDE, "reflecting")
        t_eval = np.linspace(0.0, T_MAX, N_EVAL)
        C0 = np.zeros(N)
        C0[0] = SRC_CONC
        t_ev1 = t_eval[t_eval <= SRC_OFF]
        if len(t_ev1) == 0 or t_ev1[-1] < SRC_OFF:
            t_ev1 = np.append(t_ev1, SRC_OFF)
        sol1 = solve_ivp(model_clamped, [0.0, SRC_OFF], C0, method=method,
                         t_eval=t_ev1, rtol=rtol, atol=atol, max_step=max_step)
        if not sol1.success:
            raise RuntimeError(sol1.message)
        sol1.y[0, :] = SRC_CONC
        C1 = np.clip(sol1.y[:, -1], 0.0, 1e4)
        t_ev2 = t_eval[t_eval >= SRC_OFF]
        if len(t_ev2) == 0 or t_ev2[0] > SRC_OFF:
            t_ev2 = np.insert(t_ev2, 0, SRC_OFF)
        sol2 = solve_ivp(model_reflecting, [SRC_OFF, T_MAX], C1, method=method,
                         t_eval=t_ev2, rtol=rtol, atol=atol, max_step=max_step)
        if not sol2.success:
            raise RuntimeError(sol2.message)
        start2 = 1 if (len(sol2.t) and np.isclose(sol2.t[0], sol1.t[-1])) else 0
        y_full = np.hstack([sol1.y, sol2.y[:, start2:]])
        return np.max(y_full, axis=1)

    print("Running protocol-matched constant-mobility clearance control...")
    const_depths = np.asarray([penetration_depth(run_constant_sim(k)) for k in k_vals], dtype=float)
    valid_const = np.isfinite(const_depths) & (const_depths > 0)
    alpha_const = np.full_like(const_depths, np.nan)
    alpha_const[valid_const] = np.gradient(np.log(const_depths[valid_const]), np.log(k_vals[valid_const]))

    # ============================================================
    # Panels E-F: equal-input pulse packaging
    # ============================================================
    EF = P["panels_E_F"]
    L2 = float(EF["domain_length_um"])
    N2 = int(EF["grid_points"])
    DISTAL_X = float(EF["distal_position_um"])
    RESTING_CAMP = float(EF["resting_cAMP_uM"])
    T_END = float(EF["simulation_end_s"])
    I_TOTAL = float(EF["total_integrated_input_uM_um"])
    K_PDE_PULSE = float(EF["kPDE_per_s"])
    PULSE_NUMBERS = [int(v) for v in EF["pulse_numbers"]]
    JIN_FIXED = float(EF["fixed_input_flux_uM_um_per_s"])
    T_FIRST = float(EF["first_pulse_window_start_s"])
    T_LAST = float(EF["pulse_schedule_window_end_s"])
    MIN_TAU = float(EF["minimum_pulse_duration_s"])

    s2 = np.linspace(0.0, L2, N2)
    ds2 = s2[1] - s2[0]
    distal_i = int(np.argmin(np.abs(s2 - DISTAL_X)))

    def make_pulse_schedule(I_total, Np, Jin=JIN_FIXED):
        tau = I_total / (Np * Jin)
        if tau < MIN_TAU:
            return None, None, np.nan
        window = T_LAST - T_FIRST
        interval = window / Np
        if tau > interval:
            return None, None, np.nan
        starts = T_FIRST + np.arange(Np) * interval
        ends = starts + tau
        return starts, ends, tau

    def source_on(t, starts, ends):
        return bool(np.any((t >= starts) & (t < ends)))

    def run_pulse_protocol(Np):
        starts, ends, tau = make_pulse_schedule(I_TOTAL, Np)
        if starts is None:
            raise ValueError(f"Invalid pulse schedule for N={Np}")

        def rhs(t, C):
            C = np.clip(C, 1e-12, 1e4)
            D_left = deff(C[:-1])
            D_right = deff(C[1:])
            D_mid = 2.0 * D_left * D_right / (D_left + D_right + 1e-30)
            flux = D_mid * (C[1:] - C[:-1]) / ds2

            dC = np.zeros_like(C)
            dC[1:-1] = (flux[1:] - flux[:-1]) / ds2

            src = 1.0 if source_on(t, starts, ends) else 0.0

            # Boundary nodes are half-width control volumes.
            dC[0] = 2.0 * flux[0] / ds2 + 2.0 * (JIN_FIXED / ds2) * src
            dC[-1] = -2.0 * flux[-1] / ds2

            dC -= K_PDE_PULSE * C
            return dC

        C0 = np.full(N2, RESTING_CAMP)
        t_eval = np.linspace(0.0, T_END, 700)
        sol = solve_ivp(
            rhs,
            [0.0, T_END],
            C0,
            t_eval=t_eval,
            method=method,
            rtol=rtol,
            atol=atol,
            max_step=max_step,
        )
        if (not sol.success) or np.any(~np.isfinite(sol.y)):
            raise RuntimeError(sol.message)

        Cdist = sol.y[distal_i, :]
        Ddist = deff(Cdist)
        E_relay = np.trapezoid(Ddist, sol.t)
        return {
            "t": sol.t,
            "Ddist": Ddist,
            "E_relay": E_relay,
            "starts": starts,
            "ends": ends,
            "tau": tau,
        }

    print("Running equal-input pulse fractionation traces...")
    pulse_results = {Np: run_pulse_protocol(Np) for Np in PULSE_NUMBERS}

    # ============================================================
    # Write quantitative metrics
    # ============================================================
    metrics_path = args.outdir / "Fig2_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric_group", "condition", "value", "units"])
        writer.writerow(["shared", "c_max", f"{CMAX:.9g}", "uM"])
        writer.writerow(["baseline", "penetration_depth_kPDE=0.1", f"{baseline_depth:.9g}", "um"])
        writer.writerow(["constant_mobility_control", "D_const=D_relay(c_max)", f"{D_CONST:.9g}", "um^2/s"])
        for k, depth, alpha, cdepth, calpha in zip(k_vals, depths, alpha_struct, const_depths, alpha_const):
            writer.writerow(["penetration_depth", f"kPDE={k:.9g}", f"{depth:.9g}", "um"])
            writer.writerow(["local_loglog_slope", f"structured_kPDE={k:.9g}", f"{alpha:.9g}", "dimensionless"])
            writer.writerow(["constantD_penetration_depth", f"kPDE={k:.9g}", f"{cdepth:.9g}", "um"])
            writer.writerow(["constantD_local_loglog_slope", f"kPDE={k:.9g}", f"{calpha:.9g}", "dimensionless"])
        for Np in PULSE_NUMBERS:
            out = pulse_results[Np]
            writer.writerow(["pulse_duration", f"N={Np}", f"{out['tau']:.9g}", "s"])
            writer.writerow(["distal_relay_exposure", f"N={Np}", f"{out['E_relay']:.9g}", "um^2"])

    # ============================================================
    # Figure layout
    # ============================================================
    fig_cfg = P["figure"]
    fig = plt.figure(figsize=tuple(fig_cfg["figsize_inches"]))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.54, wspace=0.42)
    axes = [fig.add_subplot(gs[i // 2, i % 2]) for i in range(6)]

    panel_titles = [
        "A. Spatiotemporal cAMP dynamics",
        "B. Propagating cAMP profiles",
        "C. Penetration depth vs PDE clearance",
        "D. Temporal occupancy-dependent mobility",
        "E. Equal-input pulse protocols",
        "F. Distal relay response",
    ]
    for ax, title in zip(axes, panel_titles):
        ax.set_title(title, fontweight="bold", loc="left", pad=4)

    # Panel A
    ax = axes[0]
    smask = s <= X_MAX_PLOT
    C_mat = y_base[smask, :]
    img = ax.imshow(
        C_mat,
        aspect="auto",
        origin="lower",
        extent=[t_base[0], t_base[-1], s[smask][0], s[smask][-1]],
        cmap="inferno",
        vmin=0,
        vmax=SRC_CONC,
    )
    cb = fig.colorbar(img, ax=ax, pad=0.02, fraction=0.046)
    cb.set_label(r"cAMP ($\mu$M)")
    cb.ax.tick_params(labelsize=6)
    cont = ax.contour(
        t_base,
        s[smask],
        C_mat,
        levels=[CMAX],
        colors="white",
        linestyles="dashed",
        linewidths=1.0,
    )
    ax.clabel(
        cont,
        fmt=lambda v: rf"$c_{{\max}}={CMAX:.3f}\,\mu$M",
        inline=True,
        fontsize=6,
        colors="white",
    )
    y_bar_A = X_MAX_PLOT * 0.94
    ax.plot([0, SRC_OFF], [y_bar_A, y_bar_A], color="white", lw=2.2, solid_capstyle="butt")
    ax.text(SRC_OFF / 2, y_bar_A - X_MAX_PLOT * 0.06, "1 s pulse", color="white", fontsize=6,
            ha="center", va="top")
    ax.set_xlim(0, T_MAX_PLOT)
    ax.set_ylim(0, X_MAX_PLOT)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"RI-network coordinate, $s$ ($\mu$m)")

    # Panel B
    ax = axes[1]
    profile_times = [float(v) for v in A["profile_times_s"]]
    profile_colors = ["#e63946", "#1a3a6e", "#2a9d8f", "#7fb254"]
    for t_target, col in zip(profile_times, profile_colors):
        idx = int(np.argmin(np.abs(t_base - t_target)))
        ax.plot(s[smask], y_base[smask, idx], color=col, lw=1.6, label=f"{t_target:.1f} s")
    ax.set_xlim(0, X_MAX_PLOT)
    ax.set_ylim(0, SRC_CONC * 1.1)
    ax.set_xlabel(r"RI-network coordinate, $s$ ($\mu$m)")
    ax.set_ylabel(r"cAMP ($\mu$M)")
    ax.yaxis.grid(True, color="#eeeeee", linewidth=0.6)
    ax.legend(framealpha=0.95, loc="upper right", ncol=1)

    # Panel C
    ax = axes[2]
    ax.plot(k_vals, depths, color="0.55", lw=1.0, zorder=2)
    ax.scatter(k_vals, depths, color="#7b2d8b", edgecolors="#5c0a9a", s=24, zorder=3)
    ax.axvline(K_BASE, color="#bbbbbb", lw=1.0, ls="--")
    ax.text(
        K_BASE * 1.1,
        depths.min() * 1.04,
        "Baseline\n0.1 s$^{-1}$",
        fontsize=6,
        color="#555555",
        style="italic",
        va="bottom",
    )
    ax.set_xscale("log")
    ax.set_xlim(k_vals[0] * 0.75, k_vals[-1] * 1.3)
    ax.set_ylim(depths.min() * 0.85, depths.max() * 1.20)
    ax.set_xlabel(r"Effective PDE-mediated clearance, $k_{PDE}$ (s$^{-1}$)")
    ax.set_ylabel(r"Penetration depth along $s$ ($\mu$m)")
    ax.yaxis.grid(True, color="#eeeeee", linewidth=0.6)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.0f}"))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))

    # Panel D
    ax = axes[3]
    positions = [float(v) for v in A["mobility_trace_positions_um"]]
    colors_d = ["#2a9d8f", "#7fb254", "#e8921a", "#9b30c8"]
    peak_D = 0.0
    for su, col in zip(positions, colors_d):
        li = int(np.argmin(np.abs(s - su)))
        yy = deff(y_base[li, :])
        peak_D = max(peak_D, float(np.max(yy)))
        ax.plot(t_base, yy, color=col, lw=1.6, label=rf"$s={su:g}\,\mu$m")
    ax.axvspan(0, SRC_OFF, color="0.85", alpha=0.35, lw=0)
    y_bar_D = peak_D * 1.08
    ax.plot([0, SRC_OFF], [y_bar_D, y_bar_D], color="0.15", lw=2.0, solid_capstyle="butt")
    ax.text(
        0.03,
        0.92,
        "1 s pulse",
        color="0.15",
        fontsize=6,
        ha="left",
        va="top",
        transform=ax.transAxes,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.55, pad=0.8),
    )
    ax.set_xlim(0, T_MAX)
    ax.set_ylim(0, peak_D * 1.15)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"$D_{relay}$ ($\mu$m$^2$ s$^{-1}$)")
    ax.yaxis.grid(True, color="#eeeeee", linewidth=0.6)
    ax.legend(framealpha=0.95, loc="upper right")

    # Panel E
    ax = axes[4]
    protocol_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    protocol_ls = ["-", "--", ":", "-."]
    y_positions = list(range(len(PULSE_NUMBERS), 0, -1))
    y_labels = []
    for y, Np, col in zip(y_positions, PULSE_NUMBERS, protocol_colors):
        out = pulse_results[Np]
        y_labels.append(rf"$N={Np}$, $\tau={out['tau']:.3g}$ s")
        for start, end in zip(out["starts"], out["ends"]):
            ax.plot([start, end], [y, y], color=col, lw=5, solid_capstyle="butt")
    ax.set_xlim(0, T_END)
    ax.set_ylim(0.5, len(PULSE_NUMBERS) + 0.6)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Protocol")
    ax.grid(axis="x", alpha=0.25)
    ax.text(
        0.02,
        0.94,
        rf"Same total input: $I_{{total}}={I_TOTAL:.0f}\,\mu$M $\mu$m",
        transform=ax.transAxes,
        va="top",
        fontsize=6,
        bbox=dict(facecolor="white", edgecolor="0.75", alpha=0.9),
    )

    # Panel F
    ax = axes[5]
    for Np, col, ls in zip(PULSE_NUMBERS, protocol_colors, protocol_ls):
        out = pulse_results[Np]
        ax.plot(
            out["t"],
            out["Ddist"],
            color=col,
            ls=ls,
            lw=1.8,
            label=rf"$N={Np}$, $E_{{relay}}={out['E_relay']:.0f}\,\mu$m$^2$",
        )
    ax.set_xlim(0, T_END)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"Distal $D_{relay}$ ($\mu$m$^2$ s$^{-1}$)")
    ax.yaxis.grid(True, color="#eeeeee", linewidth=0.6)
    ax.legend(framealpha=0.95, loc="lower right")
    ax.text(
        0.02,
        0.94,
        rf"$k_{{PDE}}={K_PDE_PULSE:g}$ s$^{{-1}}$",
        transform=ax.transAxes,
        va="top",
        fontsize=6,
        bbox=dict(facecolor="white", edgecolor="0.75", alpha=0.9),
    )

    fig.subplots_adjust(left=0.09, right=0.96, bottom=0.07, top=0.975, wspace=0.42, hspace=0.56)

    basename = fig_cfg["output_basename"]
    pdf_path = args.outdir / f"{basename}.pdf"
    png_path = args.outdir / f"{basename}.png"
    dpi = int(fig_cfg["dpi"])
    fig.savefig(pdf_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")
    print(f"Saved {metrics_path}")
    print(f"Baseline A-D penetration depth at kPDE={K_BASE:g} s^-1: {baseline_depth:.3f} um")
    print("Panel F distal relay exposure:")
    for Np in PULSE_NUMBERS:
        print(f"  N={Np:>2d}: E_relay={pulse_results[Np]['E_relay']:.3f} um^2")


if __name__ == "__main__":
    main()
