"""
Figure 3. PKAc-dependent PDE feedback temporally gates repeated distal PKAc release.

Canonical implementation:
- occupancy-dependent relay mobility D_relay(c)
- dynamic activation-state fraction f_AB (dimensionless), not R_AB concentration
- fixed holoenzyme output capacity H_T for paired-pulse experiments
- local PKAc release drives diffusible free PKAc
- free PKAc activates local nondiffusing PDE*
- k_eff = k_PDE,b + k_PDE,i * PDE*
- harmonic interface averaging
- half-cell finite-volume correction at both spatial boundaries
- reflecting proximal boundary when input flux is off

Outputs are saved alongside this script.
"""

import os
import json
import csv
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec

plt.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUTDIR, exist_ok=True)

P = dict(
    # Geometry / numerics
    L=20.0,                  # um, reduced RI-network coordinate
    N=150,
    distal_s=20.0,           # um
    t_end=18.0,              # s
    resting_cAMP=0.005,      # uM

    # Relay mobility
    D0=50.0,                 # um^2/s
    K_B=0.1,                 # uM
    K_A=1.0,                 # uM

    # RI activation-state kinetics
    RT=1.0,                  # uM, total accessible RI; here used to scale output capacity
    rho_RC=4.0,              # effective regulatory-to-output-pool scaling ratio
    k_on_A=10.0,             # uM^-1 s^-1
    k_off_A=10.0,            # s^-1

    # PKAc release / diffusion
    k_rel=5.0,               # s^-1
    D_PKAc=20.0,             # um^2/s
    k_PKAc_sink=0.4,         # s^-1

    # PDE feedback
    PDE_T=0.5,               # uM
    k_act=2.0,               # uM^-1 s^-1
    k_inact=0.3,             # s^-1
    k_PDE_b=0.1,             # s^-1
    k_PDE_i=3.0,             # uM^-1 s^-1 because PDE* is in uM

    # Boundary input; units are concentration*length/time
    source_flux=10.0,        # uM um s^-1
    t_p1_start=0.0,
    t_p1_end=1.0,
    interpulse_interval=4.0, # s from end of P1 to start of P2
    rtol=1e-7,
    atol=1e-11,
    max_step=0.03,
)

P["t_p2_start"] = P["t_p1_end"] + P["interpulse_interval"]
P["t_p2_end"] = P["t_p2_start"] + (P["t_p1_end"] - P["t_p1_start"])
P["H_T"] = P["RT"] / P["rho_RC"]
P["c_max"] = np.sqrt(P["K_A"] * P["K_B"])

s = np.linspace(0.0, P["L"], P["N"])
ds = s[1] - s[0]
P["distal_idx"] = int(np.argmin(np.abs(s - P["distal_s"])))


def D_relay(c, p=P):
    c = np.clip(c, 1e-12, 1e4)
    return p["D0"] * (c / (p["K_B"] + c)) * (p["K_A"] / (p["K_A"] + c))


def input_on(t, p=P):
    return (p["t_p1_start"] <= t < p["t_p1_end"]) or (p["t_p2_start"] <= t < p["t_p2_end"])


def run_sim(k_PDE_i=None, p=P):
    pp = dict(p)
    if k_PDE_i is not None:
        pp["k_PDE_i"] = float(k_PDE_i)

    N = pp["N"]

    def rhs(t, y):
        c = np.clip(y[0*N:1*N], 1e-12, 1e4)
        fAB = np.clip(y[1*N:2*N], 0.0, 1.0)
        pkac = np.maximum(y[2*N:3*N], 0.0)
        pde = np.clip(y[3*N:4*N], 0.0, pp["PDE_T"])

        # Dynamic activation-state fraction.
        fB = (1.0 - fAB) * c / (pp["K_B"] + c)
        dfAB = pp["k_on_A"] * c * fB - pp["k_off_A"] * fAB

        # Fixed output capacity in paired-pulse experiments.
        release = pp["k_rel"] * (fAB ** 2) * pp["H_T"]

        # Local effective cAMP clearance.
        k_eff = pp["k_PDE_b"] + pp["k_PDE_i"] * pde

        # cAMP transport: harmonic interface averaging.
        Dl = D_relay(c[:-1], pp)
        Dr = D_relay(c[1:], pp)
        Dm = 2.0 * Dl * Dr / (Dl + Dr + 1e-30)
        flux_c = Dm * (c[1:] - c[:-1]) / ds

        dc = np.zeros(N)
        dc[1:-1] = (flux_c[1:] - flux_c[:-1]) / ds
        src = 1.0 if input_on(t, pp) else 0.0
        # Boundary nodes are half-width control volumes.
        dc[0] = 2.0 * flux_c[0] / ds + 2.0 * pp["source_flux"] * src / ds
        dc[-1] = -2.0 * flux_c[-1] / ds
        dc -= k_eff * c

        # Free PKAc diffusion / clearance.
        flux_p = pp["D_PKAc"] * (pkac[1:] - pkac[:-1]) / ds
        dpkac = np.zeros(N)
        dpkac[1:-1] = (flux_p[1:] - flux_p[:-1]) / ds
        dpkac[0] = 2.0 * flux_p[0] / ds
        dpkac[-1] = -2.0 * flux_p[-1] / ds
        dpkac += release - pp["k_PKAc_sink"] * pkac

        # Local, nondiffusing PDE feedback state.
        dpde = pp["k_act"] * pkac * (pp["PDE_T"] - pde) - pp["k_inact"] * pde

        return np.concatenate([dc, dfAB, dpkac, dpde])

    y0 = np.zeros(4 * N)
    y0[0:N] = pp["resting_cAMP"]

    t_eval = np.linspace(0.0, pp["t_end"], 1200)
    sol = solve_ivp(
        rhs, [0.0, pp["t_end"]], y0,
        t_eval=t_eval, method="BDF",
        rtol=pp["rtol"], atol=pp["atol"], max_step=pp["max_step"],
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol, pp


def extract(sol, p):
    N = p["N"]
    t = sol.t
    c = sol.y[0*N:1*N, :]
    fAB = sol.y[1*N:2*N, :]
    pkac = sol.y[2*N:3*N, :]
    pde = sol.y[3*N:4*N, :]
    release = p["k_rel"] * (fAB ** 2) * p["H_T"]
    return t, c, fAB, pkac, pde, release


def paired_metrics(t, c, fAB, pkac, pde, release, p):
    di = p["distal_idx"]
    rr = release[di, :]
    onset_interval = p["t_p2_start"] - p["t_p1_start"]
    w1 = (t >= p["t_p1_start"]) & (t < p["t_p2_start"])
    w2 = (t >= p["t_p2_start"]) & (t < p["t_p2_start"] + onset_interval)

    p1_peak = float(np.max(rr[w1]))
    p2_peak = float(np.max(rr[w2]))
    p1_int = float(np.trapezoid(rr[w1], t[w1]))
    p2_int = float(np.trapezoid(rr[w2], t[w2]))

    return dict(
        distal_cAMP_peak=float(np.max(c[di, :])),
        distal_fAB_peak=float(np.max(fAB[di, :])),
        distal_free_PKAc_peak=float(np.max(pkac[di, :])),
        distal_PDEstar_peak=float(np.max(pde[di, :])),
        P1_peak_release=p1_peak,
        P2_peak_release=p2_peak,
        peak_P2_P1=p2_peak / p1_peak,
        P1_integrated_release=p1_int,
        P2_integrated_release=p2_int,
        integrated_P2_P1=p2_int / p1_int,
    )


print("Running canonical Fig. 3 feedback simulation...")
sol_fb, p_fb = run_sim(P["k_PDE_i"])
t, c, fAB, pkac, pde, release = extract(sol_fb, p_fb)
metrics_fb = paired_metrics(t, c, fAB, pkac, pde, release, p_fb)

print("Running no-induced-PDE-feedback control...")
sol_ctl, p_ctl = run_sim(0.0)
t0, c0, fAB0, pkac0, pde0, release0 = extract(sol_ctl, p_ctl)
metrics_ctl = paired_metrics(t0, c0, fAB0, pkac0, pde0, release0, p_ctl)

print("\nCanonical feedback metrics:")
for k, v in metrics_fb.items():
    print(f"  {k}: {v:.6g}")
print("\nNo-induced-feedback control:")
for k, v in metrics_ctl.items():
    print(f"  {k}: {v:.6g}")

# Save parameters as JSON and human-readable TXT.
with open(os.path.join(OUTDIR, "fig03_parameters.json"), "w") as fh:
    json.dump(P, fh, indent=2)

with open(os.path.join(OUTDIR, "fig03_parameters.txt"), "w") as fh:
    fh.write("FIGURE 3 PARAMETERS - canonical paired-pulse feedback model\n")
    fh.write("=========================================================\n\n")
    for k, v in P.items():
        if k != "distal_idx":
            fh.write(f"{k} = {v}\n")
    fh.write("\nImportant implementation choices\n")
    fh.write("- f_AB is a dimensionless activation-state fraction.\n")
    fh.write("- H_T = R_T/rho_RC is fixed in Fig. 3; no cumulative output-pool depletion.\n")
    fh.write("- cAMP and PKAc use half-cell finite-volume boundary corrections.\n")
    fh.write("- Proximal boundary is reflecting whenever source_flux is off.\n")
    fh.write("- Harmonic averaging is used for D_relay at cAMP interfaces.\n")
    fh.write("- PDE* is local and nondiffusing.\n")
    fh.write("- k_PDE_i units are uM^-1 s^-1 because PDE* is represented in uM.\n")
    fh.write("- Peak/integrated P2/P1 use equal 5-s response windows: 0-5 s and 5-10 s.\n")

# Save metrics.
with open(os.path.join(OUTDIR, "fig03_metrics.csv"), "w", newline="") as fh:
    writer = csv.writer(fh)
    writer.writerow(["metric", "feedback_kPDEi_3", "no_induced_feedback_kPDEi_0"])
    for key in metrics_fb:
        writer.writerow([key, metrics_fb[key], metrics_ctl[key]])

# Figure
fig = plt.figure(figsize=(7.2, 6.7))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.44, wspace=0.42)
axes = [fig.add_subplot(gs[i // 2, i % 2]) for i in range(4)]

di = P["distal_idx"]

def pulse_lines(ax, light=True):
    col = "white" if light else "0.45"
    for tm in [P["t_p1_end"], P["t_p2_start"], P["t_p2_end"]]:
        ax.axvline(tm, color=col, lw=0.9, ls="--", alpha=0.65)

# A. Free cAMP
ax = axes[0]
vmax_c = min(np.max(c) * 0.85, 3.0 * P["c_max"])
im = ax.pcolormesh(t, s, c, cmap="plasma",
                   norm=mcolors.Normalize(0, vmax_c),
                   shading="auto", rasterized=True)
fig.colorbar(im, ax=ax, pad=0.02, label=r"Free cAMP ($\mu$M)")
ax.contour(t, s, c, levels=[P["c_max"]], colors="white",
           linewidths=1.1, linestyles="--", alpha=0.85)
ax.axhline(s[di], color="white", lw=0.9, ls=":", alpha=0.8)
pulse_lines(ax, light=True)
ax.set_title("A. Free cAMP", loc="left", fontweight="bold")
ax.set_xlabel("Time (s)")
ax.set_ylabel(r"RI-network coordinate, $s$ ($\mu$m)")

# B. Free PKAc
ax = axes[1]
vmax_p = np.max(pkac) * 0.90
im = ax.pcolormesh(t, s, pkac, cmap="inferno",
                   norm=mcolors.Normalize(0, vmax_p),
                   shading="auto", rasterized=True)
fig.colorbar(im, ax=ax, pad=0.02, label=r"Free PKAc ($\mu$M)")
ax.axhline(s[di], color="white", lw=0.9, ls=":", alpha=0.8)
pulse_lines(ax, light=True)
ax.set_title("B. Free PKAc", loc="left", fontweight="bold")
ax.set_xlabel("Time (s)")
ax.set_ylabel(r"RI-network coordinate, $s$ ($\mu$m)")

# C. PDE* feedback state
ax = axes[2]
vmax_d = np.max(pde) * 0.95
im = ax.pcolormesh(t, s, pde, cmap="YlOrRd",
                   norm=mcolors.Normalize(0, vmax_d),
                   shading="auto", rasterized=True)
fig.colorbar(im, ax=ax, pad=0.02, label=r"PDE* ($\mu$M)")
ax.axhline(s[di], color="0.25", lw=0.9, ls=":", alpha=0.8)
pulse_lines(ax, light=False)
ax.set_title("C. PDE* feedback state", loc="left", fontweight="bold")
ax.set_xlabel("Time (s)")
ax.set_ylabel(r"RI-network coordinate, $s$ ($\mu$m)")

# D. Distal local PKAc release
ax = axes[3]
rr = release[di, :]
P1pk = metrics_fb["P1_peak_release"]
rrn = rr / P1pk
w1 = t < P["t_p2_start"]
w2 = t >= P["t_p2_start"]
ax.plot(t[w1], rrn[w1], lw=2.2, label="P1")
ax.plot(t[w2], rrn[w2], lw=2.2, ls="--", label="P2")
ax.axvspan(P["t_p1_start"], P["t_p1_end"], alpha=0.08)
ax.axvspan(P["t_p2_start"], P["t_p2_end"], alpha=0.08)
ax.axvline(P["t_p2_start"], lw=1.0, ls=":", alpha=0.8)
ax.set_xlim(0, 11)
ax.set_ylim(0, 1.15)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Local PKAc release rate\n(normalized to P1 peak)")
ax.set_title("D. Distal local PKAc release", loc="left", fontweight="bold")
ax.legend(loc="upper left", frameon=True)
ax.text(0.98, 0.72,
        "Peak local-release P2/P1 = {:.2f}\nIntegrated local-release P2/P1 = {:.2f}\nNo induced feedback (peak) = {:.2f}".format(
            metrics_fb["peak_P2_P1"], metrics_fb["integrated_P2_P1"],
            metrics_ctl["peak_P2_P1"]),
        transform=ax.transAxes, ha="right", va="top", fontsize=6.4, bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.5))
ax.yaxis.grid(True, alpha=0.2)

fig.tight_layout()

pdf = os.path.join(OUTDIR, "fig03.pdf")
png = os.path.join(OUTDIR, "fig03.png")
fig.savefig(pdf, dpi=300, bbox_inches="tight")
fig.savefig(png, dpi=300, bbox_inches="tight")
print(f"\nSaved:\n  {pdf}\n  {png}")
