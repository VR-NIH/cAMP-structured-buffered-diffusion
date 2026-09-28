"""
Figure S5. Sensitivity of distal PKAc signaling to the structured-buffered
mobility prefactor D_relay,0.

This script repeats the canonical Fig. 3 paired-pulse simulation with
D_relay,0 = 50 or 10 um^2/s while holding all other parameters fixed.
The readout is free PKAc at s = 20 um.
"""

import os
import json
import csv
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUTDIR, exist_ok=True)

plt.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

BASE = dict(
    L=20.0,
    N=150,
    distal_s=20.0,
    t_end=18.0,
    resting_cAMP=0.005,
    D0=50.0,
    K_B=0.1,
    K_A=1.0,
    RT=1.0,
    rho_RC=4.0,
    k_on_A=10.0,
    k_off_A=10.0,
    k_rel=5.0,
    D_PKAc=20.0,
    k_PKAc_sink=0.4,
    PDE_T=0.5,
    k_act=2.0,
    k_inact=0.3,
    k_PDE_b=0.1,
    k_PDE_i=3.0,
    source_flux=10.0,
    t_p1_start=0.0,
    t_p1_end=1.0,
    interpulse_interval=4.0,
    rtol=1e-7,
    atol=1e-11,
    max_step=0.03,
)
BASE["t_p2_start"] = BASE["t_p1_end"] + BASE["interpulse_interval"]
BASE["t_p2_end"] = BASE["t_p2_start"] + (BASE["t_p1_end"] - BASE["t_p1_start"])
BASE["H_T"] = BASE["RT"] / BASE["rho_RC"]
BASE["c_max"] = float(np.sqrt(BASE["K_A"] * BASE["K_B"]))
D0_VALUES = [50.0, 10.0]


def D_relay(c, p):
    c = np.clip(c, 1e-12, 1e4)
    return p["D0"] * (c / (p["K_B"] + c)) * (p["K_A"] / (p["K_A"] + c))


def input_on(t, p):
    return ((p["t_p1_start"] <= t < p["t_p1_end"]) or
            (p["t_p2_start"] <= t < p["t_p2_end"]))


def simulate(D0):
    p = dict(BASE)
    p["D0"] = float(D0)
    N = int(p["N"])
    s = np.linspace(0.0, p["L"], N)
    ds = s[1] - s[0]
    di = int(np.argmin(np.abs(s - p["distal_s"])))

    def rhs(t, y):
        c = np.clip(y[0*N:1*N], 1e-12, 1e4)
        fAB = np.clip(y[1*N:2*N], 0.0, 1.0)
        pkac = np.maximum(y[2*N:3*N], 0.0)
        pde = np.clip(y[3*N:4*N], 0.0, p["PDE_T"])

        fB = (1.0 - fAB) * c / (p["K_B"] + c)
        dfAB = p["k_on_A"] * c * fB - p["k_off_A"] * fAB
        release = p["k_rel"] * (fAB ** 2) * p["H_T"]
        k_eff = p["k_PDE_b"] + p["k_PDE_i"] * pde

        Dl = D_relay(c[:-1], p)
        Dr = D_relay(c[1:], p)
        Dm = 2.0 * Dl * Dr / (Dl + Dr + 1e-30)
        flux_c = Dm * (c[1:] - c[:-1]) / ds
        dc = np.zeros(N)
        dc[1:-1] = (flux_c[1:] - flux_c[:-1]) / ds
        src = 1.0 if input_on(t, p) else 0.0
        dc[0] = 2.0 * flux_c[0] / ds + 2.0 * p["source_flux"] * src / ds
        dc[-1] = -2.0 * flux_c[-1] / ds
        dc -= k_eff * c

        flux_p = p["D_PKAc"] * (pkac[1:] - pkac[:-1]) / ds
        dpkac = np.zeros(N)
        dpkac[1:-1] = (flux_p[1:] - flux_p[:-1]) / ds
        dpkac[0] = 2.0 * flux_p[0] / ds
        dpkac[-1] = -2.0 * flux_p[-1] / ds
        dpkac += release - p["k_PKAc_sink"] * pkac

        dpde = p["k_act"] * pkac * (p["PDE_T"] - pde) - p["k_inact"] * pde
        return np.concatenate([dc, dfAB, dpkac, dpde])

    y0 = np.zeros(4 * N)
    y0[:N] = p["resting_cAMP"]
    t_eval = np.linspace(0.0, p["t_end"], 1200)
    sol = solve_ivp(
        rhs, [0.0, p["t_end"]], y0,
        t_eval=t_eval, method="BDF",
        rtol=p["rtol"], atol=p["atol"], max_step=p["max_step"],
    )
    if not sol.success:
        raise RuntimeError(sol.message)

    t = sol.t
    pk = sol.y[2*N + di, :]
    w1 = (t >= 0.0) & (t < 5.0)
    w2 = (t >= 6.0) & (t < 12.0)
    i1_local = np.where(w1)[0][np.argmax(pk[w1])]
    i2_local = np.where(w2)[0][np.argmax(pk[w2])]
    metrics = dict(
        D_relay_0_um2_per_s=float(D0),
        D_relay_max_um2_per_s=float(D0 * (p["c_max"]/(p["K_B"]+p["c_max"])) * (p["K_A"]/(p["K_A"]+p["c_max"]))),
        readout_s_um=float(s[di]),
        first_peak_free_PKAc_uM=float(pk[i1_local]),
        first_peak_time_s=float(t[i1_local]),
        second_peak_free_PKAc_uM=float(pk[i2_local]),
        second_peak_time_s=float(t[i2_local]),
        AUC_free_PKAc_0_18_uM_s=float(np.trapezoid(pk, t)),
    )
    return dict(t=t, pk=pk, metrics=metrics)


runs = [simulate(v) for v in D0_VALUES]

# Write parameters and metrics.
params = dict(BASE)
params["D0_values_um2_per_s"] = D0_VALUES
with open(os.path.join(OUTDIR, "figS05_parameters.json"), "w") as fh:
    json.dump(params, fh, indent=2)
with open(os.path.join(OUTDIR, "figS05_parameters.txt"), "w") as fh:
    fh.write("FIGURE S5 PARAMETERS - D_relay,0 sensitivity at s=20 um\n")
    fh.write("======================================================\n\n")
    for k, v in BASE.items():
        fh.write(f"{k} = {v}\n")
    fh.write(f"D0 values = {D0_VALUES}\n")
    fh.write("All parameters except D0 are identical to canonical Fig. 3.\n")

metric_fields = list(runs[0]["metrics"].keys())
with open(os.path.join(OUTDIR, "figS05_metrics.csv"), "w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=metric_fields)
    writer.writeheader()
    for r in runs:
        writer.writerow(r["metrics"])

# Figure.
fig, ax = plt.subplots(figsize=(7.5, 5.0))
for r in runs:
    d0 = r["metrics"]["D_relay_0_um2_per_s"]
    ax.plot(r["t"], r["pk"], lw=2.2,
            label=rf"$D_{{\rm relay,0}}={d0:g}\ \mu{{\rm m}}^2/{{\rm s}}$")
for a, b in [(BASE["t_p1_start"], BASE["t_p1_end"]),
             (BASE["t_p2_start"], BASE["t_p2_end"])]:
    ax.axvspan(a, b, alpha=0.10)
ax.set_xlabel("Time (s)")
ax.set_ylabel(r"Free PKAc at 20 $\mu$m ($\mu$M)")
ax.legend(frameon=False, loc="upper right")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, "figS05.pdf"), dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(OUTDIR, "figS05.png"), dpi=300, bbox_inches="tight")
plt.close(fig)

print(json.dumps([r["metrics"] for r in runs], indent=2))
