"""Revised Figure 5 RI-abundance scan.

Canonical Fig. 5 model:
  * D_relay depends on free cAMP only; R_T does not directly alter D_relay.
  * f_AB is a dimensionless activation-state fraction.
  * H_T = R_T/rho_RC sets PKAc-output capacity.
  * Fig. 5 includes cumulative local release Q_rel, so H_intact=max(H_T-Q_rel,0).
  * free PKAc activates local nondiffusing PDE*; k_eff=k_PDE_b+k_PDE_i*PDE*.
  * harmonic D_relay interface averaging.
  * nodal finite-volume half-cell correction at both boundaries.
  * single proximal flux pulse; proximal boundary reflects when the pulse is off.

The script also runs k_PDE_i=0 as a causal control.  If the high-R_T descending
limb is feedback driven, local release should lose its intermediate optimum and
peak distal cAMP should become R_T independent in this control.
"""
import os, csv, json
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
OUTDIR=os.path.join(os.path.dirname(os.path.abspath(__file__)),"outputs")
os.makedirs(OUTDIR, exist_ok=True)
P=dict(
    L=20.0, N=80, distal_s=20.0, t_end=18.0, resting_cAMP=0.005,
    RT_min=0.1, RT_max=100.0, N_RT=40, rho_RC=4.0,
    D0=50.0, K_B=0.1, K_A=1.0,
    k_on_A=10.0, k_off_A=10.0,
    k_rel=5.0, D_PKAc=20.0, k_PKAc_sink=0.4,
    PDE_T=0.5, k_act=2.0, k_inact=0.3, k_PDE_b=0.1, k_PDE_i=3.0,
    source_flux=10.0, pulse_start=0.0, pulse_end=1.0,
    rtol=1e-6, atol=1e-10, max_step=0.06,
)
P["c_max"]=float(np.sqrt(P["K_A"]*P["K_B"]))
RT_VALUES=np.logspace(np.log10(P["RT_min"]),np.log10(P["RT_max"]),P["N_RT"])
S=np.linspace(0.0,P["L"],P["N"]); DS=S[1]-S[0]
DI=int(np.argmin(np.abs(S-P["distal_s"])))

def D_relay(c,p=P):
    c=np.clip(c,1e-12,1e4)
    return p["D0"]*(c/(p["K_B"]+c))*(p["K_A"]/(p["K_A"]+c))

def simulate(RT,k_PDE_i):
    p=dict(P); p["RT"]=float(RT); p["H_T"]=p["RT"]/p["rho_RC"]; p["k_PDE_i"]=float(k_PDE_i)
    N=p["N"]
    def rhs(t,y):
        c=np.clip(y[0*N:1*N],1e-12,1e4)
        f=np.clip(y[1*N:2*N],0,1)
        pk=np.maximum(y[2*N:3*N],0)
        pe=np.clip(y[3*N:4*N],0,p["PDE_T"])
        q=np.maximum(y[4*N:5*N],0)
        fB=(1-f)*c/(p["K_B"]+c)
        df=p["k_on_A"]*c*fB-p["k_off_A"]*f
        Hint=np.maximum(p["H_T"]-q,0)
        rr=p["k_rel"]*(f**2)*Hint
        keff=p["k_PDE_b"]+p["k_PDE_i"]*pe
        Dl=D_relay(c[:-1],p); Dr=D_relay(c[1:],p)
        Dm=2*Dl*Dr/(Dl+Dr+1e-30)
        fc=Dm*(c[1:]-c[:-1])/DS
        dc=np.zeros(N); dc[1:-1]=(fc[1:]-fc[:-1])/DS
        src=1.0 if p["pulse_start"]<=t<p["pulse_end"] else 0.0
        dc[0]=2*fc[0]/DS+2*p["source_flux"]*src/DS
        dc[-1]=-2*fc[-1]/DS
        dc-=keff*c
        fp=p["D_PKAc"]*(pk[1:]-pk[:-1])/DS
        dpk=np.zeros(N); dpk[1:-1]=(fp[1:]-fp[:-1])/DS
        dpk[0]=2*fp[0]/DS; dpk[-1]=-2*fp[-1]/DS
        dpk+=rr-p["k_PKAc_sink"]*pk
        dpe=p["k_act"]*pk*(p["PDE_T"]-pe)-p["k_inact"]*pe
        dq=rr
        return np.concatenate([dc,df,dpk,dpe,dq])
    y0=np.zeros(5*N); y0[:N]=p["resting_cAMP"]
    te=np.linspace(0,p["t_end"],600)
    sol=solve_ivp(rhs,[0,p["t_end"]],y0,t_eval=te,method="BDF",rtol=p["rtol"],atol=p["atol"],max_step=p["max_step"])
    if not sol.success: raise RuntimeError(sol.message)
    t=sol.t; c=sol.y[0*N:1*N]; f=sol.y[1*N:2*N]; pk=sol.y[2*N:3*N]; pe=sol.y[3*N:4*N]; q=sol.y[4*N:5*N]
    Hint=np.maximum(p["H_T"]-q,0); rr=p["k_rel"]*(f**2)*Hint
    cd=c[DI]; fd=f[DI]; pd=pk[DI]; ed=pe[DI]; rd=rr[DI]; qd=q[DI]
    return dict(RT=float(RT),H_T=float(p["H_T"]),peak_release=float(rd.max()),integrated_release=float(np.trapezoid(rd,t)),
                peak_free_PKAc=float(pd.max()),auc_free_PKAc=float(np.trapezoid(pd,t)),peak_cAMP=float(cd.max()),auc_cAMP=float(np.trapezoid(cd,t)),
                peak_fAB=float(fd.max()),auc_fAB=float(np.trapezoid(fd,t)),peak_PDEstar=float(ed.max()),auc_PDEstar=float(np.trapezoid(ed,t)),
                final_Qrel=float(qd[-1]),fraction_output_released=float(qd[-1]/p["H_T"]))

def run_scan(k,label):
    print("Running",label,flush=True)
    return Parallel(n_jobs=8,backend="loky",verbose=5)(delayed(simulate)(float(rt),float(k)) for rt in RT_VALUES)

def A(rows,key): return np.array([r[key] for r in rows],float)

if __name__=="__main__":
    b=run_scan(P["k_PDE_i"],"feedback on")
    n=run_scan(0.0,"induced PDE feedback off")
    rt=A(b,"RT")
    keys=[k for k in b[0] if k not in ("RT","H_T")]
    with open(os.path.join(OUTDIR,"fig05_metrics.csv"),"w",newline="") as fh:
        w=csv.writer(fh); w.writerow(["RT_uM","H_T_uM"]+["feedback_"+k for k in keys]+["no_feedback_"+k for k in keys])
        for x,y in zip(b,n): w.writerow([x["RT"],x["H_T"]]+[x[k] for k in keys]+[y[k] for k in keys])
    bp=A(b,"peak_release"); bi=A(b,"integrated_release"); bpk=A(b,"peak_free_PKAc"); bau=A(b,"auc_free_PKAc"); bc=A(b,"peak_cAMP"); be=A(b,"peak_PDEstar")
    npk=A(n,"peak_release"); ni=A(n,"integrated_release"); nc=A(n,"peak_cAMP")
    ip=int(np.argmax(bp)); ii=int(np.argmax(bi))
    summary=dict(peak_release_optimum_RT_uM=float(rt[ip]),peak_release_max_uM_per_s=float(bp[ip]),integrated_release_optimum_RT_uM=float(rt[ii]),integrated_release_max_uM=float(bi[ii]),
                 no_feedback_peak_release_monotonic=bool(np.all(np.diff(npk)>=-1e-9)),no_feedback_integrated_release_monotonic=bool(np.all(np.diff(ni)>=-1e-9)),
                 no_feedback_peak_cAMP_min_uM=float(nc.min()),no_feedback_peak_cAMP_max_uM=float(nc.max()),baseline_peak_cAMP_RTmin_uM=float(bc[0]),baseline_peak_cAMP_RTmax_uM=float(bc[-1]),
                 baseline_peak_PDEstar_RTmax_uM=float(be[-1]),baseline_peak_free_PKAc_RTmax_uM=float(bpk[-1]))
    with open(os.path.join(OUTDIR,"fig05_summary.json"),"w") as fh: json.dump(summary,fh,indent=2)
    with open(os.path.join(OUTDIR,"fig05_parameters.json"),"w") as fh: json.dump(P,fh,indent=2)
    with open(os.path.join(OUTDIR,"fig05_parameters.txt"),"w") as fh:
        fh.write("FIGURE 5 PARAMETERS - RI abundance/output-capacity feedback scan\n\n")
        for k,v in P.items(): fh.write(f"{k} = {v}\n")
        fh.write("\nImplementation:\n- f_AB is dimensionless.\n- R_T does not enter D_relay(c).\n- H_T=R_T/rho_RC.\n- Fig. 5 includes cumulative Q_rel depletion.\n- half-cell finite-volume boundaries; reflecting when source off.\n- harmonic D_relay interface averaging.\n- no-feedback control sets k_PDE_i=0 but retains basal k_PDE_b.\n")
    fig,axs=plt.subplots(2,3,figsize=(7.4,4.8)); axs=axs.ravel()
    def setup(ax,title,ylabel):
        ax.set_xscale("log"); ax.set_xlabel(r"Total RI, $R_T$ ($\mu$M)"); ax.set_ylabel(ylabel); ax.set_title(title,loc="left",fontweight="bold"); ax.grid(alpha=.22)
    ax=axs[0]; ax.plot(rt,bp,"o-",ms=3,lw=1.5); ax.scatter(rt[ip],bp[ip],s=28,zorder=5); ax.annotate(rf"max {rt[ip]:.2g} $\mu$M",(rt[ip],bp[ip]),xytext=(8,-15),textcoords="offset points",fontsize=7,va="top"); setup(ax,"A. Peak local PKAc release",r"Release rate ($\mu$M s$^{-1}$)")
    iax=ax.inset_axes([.30,.11,.42,.32]); iax.plot(rt,npk,"--",lw=1.2); iax.set_xscale("log"); iax.set_yscale("log"); iax.set_title(r"$k_{PDE,i}=0$",fontsize=6); iax.tick_params(labelsize=5); iax.grid(alpha=.15)
    ax=axs[1]; ax.plot(rt,bi,"o-",ms=3,lw=1.5); ax.scatter(rt[ii],bi[ii],s=28,zorder=5); ax.annotate(rf"max {rt[ii]:.2g} $\mu$M",(rt[ii],bi[ii]),xytext=(8,-15),textcoords="offset points",fontsize=7,va="top"); setup(ax,"B. Integrated local PKAc release",r"Integrated release ($\mu$M)")
    iax=ax.inset_axes([.30,.11,.42,.32]); iax.plot(rt,ni,"--",lw=1.2); iax.set_xscale("log"); iax.set_yscale("log"); iax.set_title(r"$k_{PDE,i}=0$",fontsize=6); iax.tick_params(labelsize=5); iax.grid(alpha=.15)
    ax=axs[2]; ax.plot(rt,bpk,"o-",ms=3,lw=1.5); setup(ax,"C. Peak distal free PKAc",r"Free PKAc ($\mu$M)")
    ax=axs[3]; ax.plot(rt,bau,"o-",ms=3,lw=1.5); setup(ax,"D. AUC distal free PKAc",r"Free-PKAc AUC ($\mu$M s)")
    ax=axs[4]; ax.plot(rt,bc,"o-",ms=3,lw=1.5,label="feedback on"); ax.plot(rt,nc,"--",lw=1.4,label="induced feedback off"); setup(ax,"E. Peak distal free cAMP",r"Free cAMP ($\mu$M)"); ax.legend(frameon=False)
    ax=axs[5]; ax.plot(rt,be,"o-",ms=3,lw=1.5); setup(ax,r"F. Peak distal $PDE^*$",r"$PDE^*$ ($\mu$M)")
    fig.tight_layout(); fig.savefig(os.path.join(OUTDIR,"fig05.pdf"),dpi=300,bbox_inches="tight"); fig.savefig(os.path.join(OUTDIR,"fig05.png"),dpi=300,bbox_inches="tight"); plt.close(fig)
    print("SUMMARY",json.dumps(summary,indent=2),flush=True)
