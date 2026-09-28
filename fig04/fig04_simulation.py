"""
Figure 4. RI abundance, PDE feedback, and CNB-A kinetics define paired-pulse temporal filtering.

Reference implementation inherited from revised Fig. 3:
- D_relay(c) occupancy-dependent mobility
- dynamic activation-state fraction f_AB, not R_AB concentration
- fixed output capacity H_T = R_T/rho_RC in paired-pulse simulations
- no cumulative holoenzyme depletion
- diffusible free PKAc drives local nondiffusing PDE*
- k_eff = k_PDE,b + k_PDE,i * PDE*
- harmonic interface averaging for cAMP transport
- half-cell finite-volume correction at both spatial boundaries
- reflecting boundaries whenever proximal input flux is off
- R_T does NOT directly rescale D_relay

Top row: peak P2/P1 local distal PKAc-release ratio.
Bottom row: integrated local distal PKAc release for an isolated first-pulse reference.

A low-output reliability mask is applied where the isolated P1 peak is <1% of the
reference Fig. 3 P1 peak, to prevent denominator-driven P2/P1 artifacts.
"""

import os, json, csv, warnings
import numpy as np
from scipy.integrate import solve_ivp
from joblib import Parallel, delayed
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore", category=UserWarning)

plt.rcParams.update({
    "font.size": 7.5,
    "axes.titlesize": 8.5,
    "axes.labelsize": 7.5,
    "xtick.labelsize": 6.8,
    "ytick.labelsize": 6.8,
    "legend.fontsize": 6.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUTDIR, exist_ok=True)

BASE = dict(
    L=20.0,
    N=100,
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
    pulse_duration=1.0,
    interpulse_interval=4.0,
    rtol=2e-6,
    atol=1e-9,
    max_step=0.06,
)

NGRID = 16
RT_vals = np.logspace(-1.0, 1.3, NGRID)
IPI_vals = np.linspace(3.0, 10.0, NGRID)
PDE_vals = np.logspace(-1.0, 1.0, NGRID)
KOFF_vals = np.logspace(-1.0, 2.0, NGRID)

ATTENUATION_POINT = dict(RT=1.0, interpulse_interval=4.0)
ENHANCEMENT_POINT = dict(RT=0.2, k_off_A=80.0, K_A=1.0, interpulse_interval=4.0)


def D_relay(c, p):
    c = np.clip(c, 1e-12, 1e4)
    return p["D0"] * (c / (p["K_B"] + c)) * (p["K_A"] / (p["K_A"] + c))


def simulate(p_overrides=None, paired=True):
    p = dict(BASE)
    if p_overrides:
        p.update(p_overrides)
    N = int(p["N"])
    s = np.linspace(0.0, p["L"], N)
    ds = s[1] - s[0]
    di = int(np.argmin(np.abs(s - p["distal_s"])))
    p1s, p1e = 0.0, p["pulse_duration"]
    p2s = p1e + p["interpulse_interval"]
    p2e = p2s + p["pulse_duration"]
    H_T = p["RT"] / p["rho_RC"]

    def input_on(t):
        if p1s <= t < p1e:
            return True
        return paired and (p2s <= t < p2e)

    def rhs(t, y):
        c = np.clip(y[0*N:1*N], 1e-12, 1e4)
        fAB = np.clip(y[1*N:2*N], 0.0, 1.0)
        pkac = np.maximum(y[2*N:3*N], 0.0)
        pde = np.clip(y[3*N:4*N], 0.0, p["PDE_T"])

        fB = (1.0 - fAB) * c / (p["K_B"] + c)
        dfAB = p["k_on_A"] * c * fB - p["k_off_A"] * fAB
        release = p["k_rel"] * (fAB ** 2) * H_T
        k_eff = p["k_PDE_b"] + p["k_PDE_i"] * pde

        Dl = D_relay(c[:-1], p)
        Dr = D_relay(c[1:], p)
        Dm = 2.0 * Dl * Dr / (Dl + Dr + 1e-30)
        flux_c = Dm * (c[1:] - c[:-1]) / ds
        dc = np.zeros(N)
        dc[1:-1] = (flux_c[1:] - flux_c[:-1]) / ds
        src = 1.0 if input_on(t) else 0.0
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
    t_end = max(p["t_end"], p2e + 6.0)
    t_eval = np.linspace(0.0, t_end, 650)
    sol = solve_ivp(rhs, [0.0, t_end], y0, t_eval=t_eval, method="BDF",
                    rtol=p["rtol"], atol=p["atol"], max_step=p["max_step"])
    if not sol.success:
        raise RuntimeError(sol.message)
    t = sol.t
    fdist = sol.y[1*N + di, :]
    release_dist = p["k_rel"] * (fdist ** 2) * H_T
    return t, release_dist, p2s


def isolated_p1_metrics(p_overrides=None):
    t, rr, _ = simulate(p_overrides, paired=False)
    w = (t >= 0.0) & (t < 5.0)
    return float(np.max(rr[w])), float(np.trapezoid(rr[w], t[w]))


def paired_ratio(p_overrides=None, p1_peak=None):
    if p1_peak is None:
        p1_peak, _ = isolated_p1_metrics(p_overrides)
    t, rr, p2s = simulate(p_overrides, paired=True)
    w2 = (t >= p2s) & (t < p2s + 5.0)
    p2_peak = float(np.max(rr[w2]))
    return p2_peak / max(p1_peak, 1e-15), p2_peak


def reference_metrics():
    ov = {"RT": 1.0, "interpulse_interval": 4.0}
    p1pk, p1int = isolated_p1_metrics(ov)
    ratio, p2pk = paired_ratio(ov, p1pk)
    return p1pk, p1int, p2pk, ratio

print("Computing reference condition...")
BASE_P1_PEAK, BASE_P1_INT, BASE_P2_PEAK, BASE_RATIO = reference_metrics()
LOW_OUTPUT_THRESHOLD = 0.01 * BASE_P1_PEAK
print(f"Reference P1 peak = {BASE_P1_PEAK:.6g} uM/s")
print(f"Reference P2/P1 = {BASE_RATIO:.6g}")
print(f"Reliability-mask threshold = {LOW_OUTPUT_THRESHOLD:.6g} uM/s")


def row_reference(kind, val):
    ov = {"RT": float(val)} if kind == "RT" else {"k_PDE_i": float(val)}
    pk, integ = isolated_p1_metrics(ov)
    return kind, float(val), pk, integ

refs_jobs = [("RT", v) for v in RT_vals] + [("PDE", v) for v in PDE_vals]
refs_list = Parallel(n_jobs=8, backend="loky", verbose=5)(delayed(row_reference)(*j) for j in refs_jobs)
refs = {(k, v): (pk, integ) for k, v, pk, integ in refs_list}


def pair_job(group, a, b):
    if group == "g1":
        rt, ipi = float(a), float(b)
        ov = {"RT": rt, "interpulse_interval": ipi}
        p1pk, p1int = refs[("RT", rt)]
    elif group == "g2":
        kpi, ipi = float(a), float(b)
        ov = {"k_PDE_i": kpi, "interpulse_interval": ipi}
        p1pk, p1int = refs[("PDE", kpi)]
    elif group == "g3":
        koff, rt = float(a), float(b)
        kon = 10.0
        KA = koff / kon
        ov = {"k_off_A": koff, "k_on_A": kon, "K_A": KA, "RT": rt, "interpulse_interval": 4.0}
        p1pk, p1int = isolated_p1_metrics(ov)
    elif group == "g4":
        koff, rt = float(a), float(b)
        KA = 1.0
        kon = koff / KA
        ov = {"k_off_A": koff, "k_on_A": kon, "K_A": KA, "RT": rt, "interpulse_interval": 4.0}
        p1pk, p1int = isolated_p1_metrics(ov)
    else:
        raise ValueError(group)
    ratio, p2pk = paired_ratio(ov, p1pk)
    return group, float(a), float(b), ratio, p1pk, p1int, p2pk

jobs = []
for rt in RT_vals:
    for ipi in IPI_vals:
        jobs.append(("g1", rt, ipi))
for kpi in PDE_vals:
    for ipi in IPI_vals:
        jobs.append(("g2", kpi, ipi))
for koff in KOFF_vals:
    for rt in RT_vals:
        jobs.append(("g3", koff, rt))
for koff in KOFF_vals:
    for rt in RT_vals:
        jobs.append(("g4", koff, rt))

print(f"Running {len(jobs)} paired-map simulations...")
results_list = Parallel(n_jobs=8, backend="loky", verbose=10)(delayed(pair_job)(*j) for j in jobs)
results = {(g, a, b): (ratio, p1pk, p1int, p2pk) for g, a, b, ratio, p1pk, p1int, p2pk in results_list}

# Matrices.
g1_ratio = np.zeros((len(RT_vals), len(IPI_vals))); g1_p1pk=np.zeros_like(g1_ratio); g1_p1int=np.zeros_like(g1_ratio)
for i,rt in enumerate(RT_vals):
    for j,ipi in enumerate(IPI_vals): g1_ratio[i,j],g1_p1pk[i,j],g1_p1int[i,j],_=results[("g1",float(rt),float(ipi))]
g2_ratio = np.zeros((len(PDE_vals), len(IPI_vals))); g2_p1pk=np.zeros_like(g2_ratio); g2_p1int=np.zeros_like(g2_ratio)
for i,kpi in enumerate(PDE_vals):
    for j,ipi in enumerate(IPI_vals): g2_ratio[i,j],g2_p1pk[i,j],g2_p1int[i,j],_=results[("g2",float(kpi),float(ipi))]
g3_ratio = np.zeros((len(KOFF_vals), len(RT_vals))); g3_p1pk=np.zeros_like(g3_ratio); g3_p1int=np.zeros_like(g3_ratio)
for i,koff in enumerate(KOFF_vals):
    for j,rt in enumerate(RT_vals): g3_ratio[i,j],g3_p1pk[i,j],g3_p1int[i,j],_=results[("g3",float(koff),float(rt))]
g4_ratio = np.zeros((len(KOFF_vals), len(RT_vals))); g4_p1pk=np.zeros_like(g4_ratio); g4_p1int=np.zeros_like(g4_ratio)
for i,koff in enumerate(KOFF_vals):
    for j,rt in enumerate(RT_vals): g4_ratio[i,j],g4_p1pk[i,j],g4_p1int[i,j],_=results[("g4",float(koff),float(rt))]

# Exact low-RT limiting enhancement point.
enh_ov={"RT":0.2,"interpulse_interval":4.0,"k_off_A":80.0,"k_on_A":80.0,"K_A":1.0}
enh_p1pk,enh_p1int=isolated_p1_metrics(enh_ov); enh_ratio,enh_p2pk=paired_ratio(enh_ov,enh_p1pk)

params=dict(BASE)
params.update({"NGRID":NGRID,"RT_range_uM":[float(RT_vals[0]),float(RT_vals[-1])],"IPI_range_s":[float(IPI_vals[0]),float(IPI_vals[-1])],"k_PDE_i_range_uM^-1_s^-1":[float(PDE_vals[0]),float(PDE_vals[-1])],"k_off_A_range_s^-1":[float(KOFF_vals[0]),float(KOFF_vals[-1])],"low_output_mask_fraction_of_reference_P1_peak":0.01,"reference_peak_P2_P1":BASE_RATIO,"low_RT_enhancement_peak_P2_P1":enh_ratio})
with open(os.path.join(OUTDIR,"fig04_parameters.json"),"w") as fh: json.dump(params,fh,indent=2)
with open(os.path.join(OUTDIR,"fig04_parameters.txt"),"w") as fh:
    fh.write("FIGURE 4 PARAMETERS - reference paired-pulse phase maps\n======================================================\n\n")
    for k,v in BASE.items(): fh.write(f"{k} = {v}\n")
    fh.write(f"\nNGRID = {NGRID}\nRT sweep = {RT_vals[0]:g} to {RT_vals[-1]:g} uM (log-spaced)\nInterpulse interval sweep = {IPI_vals[0]:g} to {IPI_vals[-1]:g} s\nk_PDE_i sweep = {PDE_vals[0]:g} to {PDE_vals[-1]:g} uM^-1 s^-1 (log-spaced)\nk_off_A sweep = {KOFF_vals[0]:g} to {KOFF_vals[-1]:g} s^-1 (log-spaced)\n")
    fh.write("\nSweep 3: k_on_A fixed at 10 uM^-1 s^-1; K_A=k_off_A/k_on_A and the same K_A is used in D_relay.\nSweep 4: K_A fixed at 1 uM; k_on_A=k_off_A/K_A.\nP1 denominator and bottom-row output come from a matched isolated first-pulse simulation.\nP1 integrated local distal release is integrated over 0-5 s and has units uM.\nP2 peak is measured over the 5 s following P2 onset.\nValues with isolated P1 peak <1% of the reference Fig. 3 P1 peak are hatched.\nFilled star: reference Fig. 3 attenuation point.\nOpen star: low-RT limiting enhancement example (RT=0.2 uM, k_off,A=80 s^-1, K_A=1 uM, IPI=4 s).\n")
    fh.write(f"Reference ratio = {BASE_RATIO:.6g}\nLow-RT enhancement ratio = {enh_ratio:.6g}\n")

with open(os.path.join(OUTDIR,"fig04_summary_metrics.csv"),"w",newline="") as fh:
    w=csv.writer(fh);w.writerow(["metric","value"]);w.writerows([
        ["reference_P1_peak_release_uM_per_s",BASE_P1_PEAK],["reference_P1_integrated_release_uM",BASE_P1_INT],["reference_P2_peak_release_uM_per_s",BASE_P2_PEAK],["reference_peak_P2_P1",BASE_RATIO],["low_RT_example_P1_peak_release_uM_per_s",enh_p1pk],["low_RT_example_P1_integrated_release_uM",enh_p1int],["low_RT_example_P2_peak_release_uM_per_s",enh_p2pk],["low_RT_example_peak_P2_P1",enh_ratio],
        ["g1_ratio_min",np.nanmin(g1_ratio)],["g1_ratio_max",np.nanmax(g1_ratio)],["g1_fraction_gt1",np.mean(g1_ratio>1)],
        ["g2_ratio_min",np.nanmin(g2_ratio)],["g2_ratio_max",np.nanmax(g2_ratio)],["g2_fraction_gt1",np.mean(g2_ratio>1)],
        ["g3_ratio_min",np.nanmin(g3_ratio)],["g3_ratio_max",np.nanmax(g3_ratio)],["g3_fraction_gt1",np.mean(g3_ratio>1)],
        ["g4_ratio_min",np.nanmin(g4_ratio)],["g4_ratio_max",np.nanmax(g4_ratio)],["g4_fraction_gt1",np.mean(g4_ratio>1)],])

with open(os.path.join(OUTDIR,"fig04_grid_metrics.csv"),"w",newline="") as fh:
    w=csv.writer(fh);w.writerow(["sweep","x","y","peak_P2_P1","P1_peak_release_uM_per_s","P1_integrated_release_uM","low_output_mask"])
    for i,rt in enumerate(RT_vals):
        for j,ipi in enumerate(IPI_vals): w.writerow(["RT_x_IPI",ipi,rt,g1_ratio[i,j],g1_p1pk[i,j],g1_p1int[i,j],g1_p1pk[i,j]<LOW_OUTPUT_THRESHOLD])
    for i,kpi in enumerate(PDE_vals):
        for j,ipi in enumerate(IPI_vals): w.writerow(["kPDEi_x_IPI",ipi,kpi,g2_ratio[i,j],g2_p1pk[i,j],g2_p1int[i,j],g2_p1pk[i,j]<LOW_OUTPUT_THRESHOLD])
    for i,koff in enumerate(KOFF_vals):
        for j,rt in enumerate(RT_vals):
            w.writerow(["koff_x_RT_KA_varies",rt,koff,g3_ratio[i,j],g3_p1pk[i,j],g3_p1int[i,j],g3_p1pk[i,j]<LOW_OUTPUT_THRESHOLD]);w.writerow(["koff_x_RT_KA_fixed",rt,koff,g4_ratio[i,j],g4_p1pk[i,j],g4_p1int[i,j],g4_p1pk[i,j]<LOW_OUTPUT_THRESHOLD])

fig=plt.figure(figsize=(11.8,5.9));gs=gridspec.GridSpec(2,4,figure=fig,hspace=.48,wspace=.50)
sweeps=[
 dict(ratio=g1_ratio,p1pk=g1_p1pk,p1int=g1_p1int,xv=IPI_vals,yv=RT_vals,xlabel="Interpulse interval (s)",ylabel=r"$R_T$ ($\mu$M)",xscale="linear",yscale="log",title=r"A. $R_T$ vs interpulse interval",att_star=(4.,1.),enh_star=None),
 dict(ratio=g2_ratio,p1pk=g2_p1pk,p1int=g2_p1int,xv=IPI_vals,yv=PDE_vals,xlabel="Interpulse interval (s)",ylabel=r"$k_{\mathrm{PDE,i}}$ ($\mu$M$^{-1}$ s$^{-1}$)",xscale="linear",yscale="log",title=r"B. PDE feedback vs interpulse interval",att_star=(4.,3.),enh_star=None),
 dict(ratio=g3_ratio,p1pk=g3_p1pk,p1int=g3_p1int,xv=RT_vals,yv=KOFF_vals,xlabel=r"$R_T$ ($\mu$M)",ylabel=r"$k_{\mathrm{off,A}}$ (s$^{-1}$)",xscale="log",yscale="log",title=r"C. $k_{\mathrm{off,A}}$ vs $R_T$ ($K_A$ varies)",att_star=(1.,10.),enh_star=None),
 dict(ratio=g4_ratio,p1pk=g4_p1pk,p1int=g4_p1int,xv=RT_vals,yv=KOFF_vals,xlabel=r"$R_T$ ($\mu$M)",ylabel=r"$k_{\mathrm{off,A}}$ (s$^{-1}$)",xscale="log",yscale="log",title=r"D. $k_{\mathrm{off,A}}$ vs $R_T$ ($K_A=1\ \mu$M)",att_star=(1.,10.),enh_star=(.2,80.)),]
ratio_vmin=max(0.,min(np.nanmin(s["ratio"]) for s in sweeps));ratio_vmax=2.;ratio_norm=mcolors.TwoSlopeNorm(vmin=ratio_vmin,vcenter=1.,vmax=ratio_vmax)
pos_vals=np.concatenate([s["p1int"][s["p1int"]>0] for s in sweeps]);int_vmin=max(np.nanpercentile(pos_vals,2),1e-6);int_vmax=np.nanpercentile(pos_vals,98);int_norm=mcolors.LogNorm(vmin=int_vmin,vmax=int_vmax)
for col,sw in enumerate(sweeps):
    ax=fig.add_subplot(gs[0,col]);im=ax.pcolormesh(sw["xv"],sw["yv"],np.minimum(sw["ratio"],ratio_vmax),cmap="RdBu_r",norm=ratio_norm,shading="auto",rasterized=True)
    ax.contour(sw["xv"],sw["yv"],sw["ratio"],levels=[1.],colors="black",linewidths=1.2,linestyles="--")
    low=sw["p1pk"]<LOW_OUTPUT_THRESHOLD
    if np.any(low): ax.contourf(sw["xv"],sw["yv"],low.astype(float),levels=[.5,1.5],colors="none",hatches=["////"])
    if sw["att_star"] is not None: ax.plot(*sw["att_star"],marker="*",ms=9,mfc="black",mec="black",zorder=6)
    if sw["enh_star"] is not None: ax.plot(*sw["enh_star"],marker="*",ms=10,mfc="white",mec="black",mew=1.2,zorder=7)
    ax.set_xscale(sw["xscale"]);ax.set_yscale(sw["yscale"]);ax.set_xlabel(sw["xlabel"]);ax.set_ylabel(sw["ylabel"]);ax.set_title(sw["title"],loc="left",fontweight="bold")
    cb=fig.colorbar(im,ax=ax,pad=.018,extend="max");cb.set_label("Peak $P_2/P_1$")
    ax2=fig.add_subplot(gs[1,col]);im2=ax2.pcolormesh(sw["xv"],sw["yv"],np.maximum(sw["p1int"],int_vmin),cmap="viridis",norm=int_norm,shading="auto",rasterized=True)
    ax2.set_xscale(sw["xscale"]);ax2.set_yscale(sw["yscale"]);ax2.set_xlabel(sw["xlabel"]);ax2.set_ylabel(sw["ylabel"]);ax2.set_title("P1 integrated local release",loc="left",fontweight="bold")
    cb2=fig.colorbar(im2,ax=ax2,pad=.018);cb2.set_label(r"$\int R_{\mathrm{PKAc}}dt$ ($\mu$M)")
fig.text(.01,.006,"Dashed contour: P2/P1 = 1.  Filled star: reference Fig. 3 attenuation condition.  Open star: low-$R_T$ limiting enhancement example (Fig. S6).  Hatched: P1 peak <1% of reference.",fontsize=7.2,ha="left",va="bottom")
fig.subplots_adjust(left=.065,right=.985,top=.965,bottom=.11)
pdf=os.path.join(OUTDIR,"fig04.pdf");png=os.path.join(OUTDIR,"fig04.png");fig.savefig(pdf,dpi=300,bbox_inches="tight");fig.savefig(png,dpi=300,bbox_inches="tight");plt.close(fig)
print("\nSummary:")
print(f"reference peak P2/P1 = {BASE_RATIO:.4f}")
print(f"low-RT limiting example peak P2/P1 = {enh_ratio:.4f}")
for name,arr in [("A RT x IPI",g1_ratio),("B PDE x IPI",g2_ratio),("C koff x RT, KA varies",g3_ratio),("D koff x RT, KA fixed",g4_ratio)]: print(f"{name}: min={np.nanmin(arr):.3f}, max={np.nanmax(arr):.3f}, frac>1={np.mean(arr>1):.3f}")
print("Saved",pdf,png)
