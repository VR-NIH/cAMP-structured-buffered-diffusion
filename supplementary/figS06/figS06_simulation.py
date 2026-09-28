import os, json, csv
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUTDIR, exist_ok=True)

plt.rcParams.update({
    'font.size': 7.5,
    'axes.titlesize': 8.5,
    'axes.labelsize': 7.5,
    'xtick.labelsize': 6.8,
    'ytick.labelsize': 6.8,
    'legend.fontsize': 6.5,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

P = dict(
    L=20.0,
    N=100,  # match revised Fig. 4 phase maps exactly
    distal_s=20.0,
    t_end=18.0,
    resting_cAMP=0.005,
    D0=50.0,
    K_B=0.1,
    K_A=1.0,
    RT=0.2,
    rho_RC=4.0,
    k_off_A=80.0,
    k_on_A=80.0,  # fixed K_A = koff/kon = 1 uM
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


def D_relay(c, p=P):
    c = np.clip(c, 1e-12, 1e4)
    return p['D0'] * (c/(p['K_B']+c)) * (p['K_A']/(p['K_A']+c))


def simulate(p, paired=True):
    N = int(p['N'])
    s = np.linspace(0, p['L'], N)
    ds = s[1] - s[0]
    di = int(np.argmin(np.abs(s-p['distal_s'])))
    p1s, p1e = 0.0, p['pulse_duration']
    p2s = p1e + p['interpulse_interval']
    p2e = p2s + p['pulse_duration']
    H_T = p['RT']/p['rho_RC']

    def input_on(t):
        if p1s <= t < p1e:
            return True
        return paired and (p2s <= t < p2e)

    def rhs(t, y):
        c = np.clip(y[0*N:1*N], 1e-12, 1e4)
        fAB = np.clip(y[1*N:2*N], 0.0, 1.0)
        pkac = np.maximum(y[2*N:3*N], 0.0)
        pde = np.clip(y[3*N:4*N], 0.0, p['PDE_T'])

        fB = (1.0-fAB) * c/(p['K_B']+c)
        dfAB = p['k_on_A']*c*fB - p['k_off_A']*fAB
        release = p['k_rel']*(fAB**2)*H_T
        k_eff = p['k_PDE_b'] + p['k_PDE_i']*pde

        Dl = D_relay(c[:-1], p); Dr = D_relay(c[1:], p)
        Dm = 2.0*Dl*Dr/(Dl+Dr+1e-30)
        flux_c = Dm*(c[1:]-c[:-1])/ds
        dc = np.zeros(N)
        dc[1:-1] = (flux_c[1:]-flux_c[:-1])/ds
        src = 1.0 if input_on(t) else 0.0
        dc[0] = 2.0*flux_c[0]/ds + 2.0*p['source_flux']*src/ds
        dc[-1] = -2.0*flux_c[-1]/ds
        dc -= k_eff*c

        flux_p = p['D_PKAc']*(pkac[1:]-pkac[:-1])/ds
        dpkac = np.zeros(N)
        dpkac[1:-1] = (flux_p[1:]-flux_p[:-1])/ds
        dpkac[0] = 2.0*flux_p[0]/ds
        dpkac[-1] = -2.0*flux_p[-1]/ds
        dpkac += release - p['k_PKAc_sink']*pkac

        dpde = p['k_act']*pkac*(p['PDE_T']-pde) - p['k_inact']*pde
        return np.concatenate([dc, dfAB, dpkac, dpde])

    y0 = np.zeros(4*N)
    y0[:N] = p['resting_cAMP']
    t_eval = np.linspace(0, p['t_end'], 800)
    sol = solve_ivp(rhs, [0, p['t_end']], y0, t_eval=t_eval, method='BDF',
                    rtol=p['rtol'], atol=p['atol'], max_step=p['max_step'])
    if not sol.success:
        raise RuntimeError(sol.message)

    t = sol.t
    C = sol.y[0*N:1*N, :]
    fAB = sol.y[1*N:2*N, :]
    PKAc = sol.y[2*N:3*N, :]
    PDE = sol.y[3*N:4*N, :]
    release = p['k_rel']*(fAB**2)*H_T
    return dict(t=t,s=s,C=C,fAB=fAB,PKAc=PKAc,PDE=PDE,release=release,
                distal_i=di,p1s=p1s,p1e=p1e,p2s=p2s,p2e=p2e,H_T=H_T)

paired = simulate(P, paired=True)
isolated = simulate(P, paired=False)

# canonical Fig. 3 reference for relative output magnitude
P_can = dict(P)
P_can.update(dict(RT=1.0,k_off_A=10.0,k_on_A=10.0,K_A=1.0))
canonical = simulate(P_can, paired=False)

# Metrics using same definition as revised Fig. 4: paired P2 peak / matched isolated P1 peak.
t = paired['t']; di = paired['distal_i']
rr = paired['release'][di,:]
rr_iso = isolated['release'][isolated['distal_i'],:]
rr_can = canonical['release'][canonical['distal_i'],:]

w1_iso = (isolated['t'] >= 0.0) & (isolated['t'] < 5.0)
w2 = (t >= paired['p2s']) & (t < paired['p2s'] + 5.0)
w1_pair = (t >= 0.0) & (t < 5.0)

p1_peak_iso = float(np.max(rr_iso[w1_iso]))
p1_peak_pair = float(np.max(rr[w1_pair]))
p2_peak = float(np.max(rr[w2]))
peak_ratio = p2_peak/max(p1_peak_iso,1e-15)
p1_int = float(np.trapezoid(rr_iso[w1_iso], isolated['t'][w1_iso]))
p2_int = float(np.trapezoid(rr[w2], t[w2]))
int_ratio = p2_int/max(p1_int,1e-15)
can_p1_peak = float(np.max(rr_can[(canonical['t']>=0)&(canonical['t']<5)]))
relative_output = p1_peak_iso/max(can_p1_peak,1e-15)

# Other distal metrics.
Cdist = paired['C'][di,:]
PKdist = paired['PKAc'][di,:]
PDEdist = paired['PDE'][di,:]
metrics = {
    'RT_uM': P['RT'],
    'K_A_uM': P['K_A'],
    'k_on_A_uM^-1_s^-1': P['k_on_A'],
    'k_off_A_s^-1': P['k_off_A'],
    'interpulse_interval_s': P['interpulse_interval'],
    'distal_s_um': P['distal_s'],
    'isolated_P1_peak_release_uM_s^-1': p1_peak_iso,
    'paired_P1_peak_release_uM_s^-1': p1_peak_pair,
    'paired_P2_peak_release_uM_s^-1': p2_peak,
    'peak_P2_over_P1': peak_ratio,
    'isolated_P1_integrated_release_uM': p1_int,
    'paired_P2_integrated_release_uM': p2_int,
    'integrated_P2_over_P1': int_ratio,
    'P1_peak_fraction_of_canonical_Fig3': relative_output,
    'peak_distal_cAMP_uM': float(np.max(Cdist)),
    'peak_distal_free_PKAc_uM': float(np.max(PKdist)),
    'peak_distal_PDEstar_uM': float(np.max(PDEdist)),
}

# Save parameters and metrics.
with open(os.path.join(OUTDIR,'figS06_parameters.json'),'w') as f:
    json.dump(P,f,indent=2)
with open(os.path.join(OUTDIR,'figS06_parameters.txt'),'w') as f:
    f.write('Figure S6 low-RT limiting facilitation example\n')
    f.write('Canonical equations: same as revised Figs. 3-4\n')
    f.write('Readout is s=20 um to match the open-star condition in revised Fig. 4.\n\n')
    for k,v in P.items():
        f.write(f'{k}: {v}\n')
    f.write('\nDerived:\n')
    f.write(f'H_T_uM: {paired["H_T"]}\n')
    f.write(f'c_max_uM: {np.sqrt(P["K_A"]*P["K_B"])}\n')
    f.write('\nNumerical scheme:\n')
    f.write('harmonic averaging for D_relay interfaces\n')
    f.write('half-cell finite-volume correction at both boundaries\n')
    f.write('proximal imposed flux only during pulses; reflecting otherwise\n')
    f.write('distal reflecting boundary\n')
    f.write('fixed H_T; no cumulative holoenzyme depletion\n')
with open(os.path.join(OUTDIR,'figS06_metrics.csv'),'w',newline='') as f:
    w=csv.writer(f); w.writerow(['metric','value'])
    for k,v in metrics.items(): w.writerow([k,v])

# Figure
fig, axs = plt.subplots(2,2,figsize=(7.2,5.7))
axA,axB,axC,axD = axs.ravel()
extent=[t[0],t[-1],paired['s'][0],paired['s'][-1]]

im=axA.imshow(paired['C'],origin='lower',aspect='auto',extent=extent,cmap='viridis')
axA.set_title('A. Free cAMP',loc='left',fontweight='bold')
axA.set_ylabel(r'RI-network coordinate, $s$ ($\mu$m)')
axA.set_xlabel('Time (s)')
fig.colorbar(im,ax=axA,label=r'cAMP ($\mu$M)',fraction=0.046,pad=0.04)

im=axB.imshow(paired['PKAc'],origin='lower',aspect='auto',extent=extent,cmap='magma')
axB.set_title('B. Free PKAc',loc='left',fontweight='bold')
axB.set_ylabel(r'RI-network coordinate, $s$ ($\mu$m)')
axB.set_xlabel('Time (s)')
fig.colorbar(im,ax=axB,label=r'PKAc ($\mu$M)',fraction=0.046,pad=0.04)

im=axC.imshow(paired['PDE'],origin='lower',aspect='auto',extent=extent,cmap='inferno')
axC.set_title(r'C. PDE$^*$ feedback state',loc='left',fontweight='bold')
axC.set_ylabel(r'RI-network coordinate, $s$ ($\mu$m)')
axC.set_xlabel('Time (s)')
fig.colorbar(im,ax=axC,label=r'PDE$^*$ ($\mu$M)',fraction=0.046,pad=0.04)

axD.plot(t,rr,lw=1.8,label='Paired-pulse response')
axD.axvspan(paired['p1s'],paired['p1e'],alpha=0.10,label='Input pulse')
axD.axvspan(paired['p2s'],paired['p2e'],alpha=0.10)
# mark peaks
p1_idx=np.where(w1_pair)[0][np.argmax(rr[w1_pair])]
p2_idx=np.where(w2)[0][np.argmax(rr[w2])]
axD.scatter([t[p1_idx],t[p2_idx]],[rr[p1_idx],rr[p2_idx]],s=20,zorder=4)
axD.annotate('P1',xy=(t[p1_idx],rr[p1_idx]),xytext=(4,4),textcoords='offset points',fontsize=7)
axD.annotate('P2',xy=(t[p2_idx],rr[p2_idx]),xytext=(4,4),textcoords='offset points',fontsize=7)
axD.text(0.98,0.96,rf'$P_2/P_{{1,\mathrm{{iso}}}}={peak_ratio:.2f}$',transform=axD.transAxes,ha='right',va='top',fontsize=8)
axD.set_title('D. Distal local PKAc release',loc='left',fontweight='bold')
axD.set_xlabel('Time (s)')
axD.set_ylabel(r'Release rate ($\mu$M s$^{-1}$)')
axD.grid(alpha=0.22)

for ax in (axA,axB,axC):
    ax.axvline(paired['p2s'],ls='--',lw=0.8,alpha=0.65)

fig.tight_layout()
pdf=os.path.join(OUTDIR,'figS06.pdf')
png=os.path.join(OUTDIR,'figS06.png')
fig.savefig(pdf,dpi=300,bbox_inches='tight')
fig.savefig(png,dpi=300,bbox_inches='tight')
plt.close(fig)

print(json.dumps(metrics,indent=2))
print(pdf)
print(png)
