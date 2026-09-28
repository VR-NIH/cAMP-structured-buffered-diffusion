"""
Revised Supplementary Figure S3: equal-input pulse fractionation parameter sweeps.

Key revisions relative to the earlier script:
  1. resting cAMP = 0.005 uM (5 nM), matching revised Fig. 2E-F;
  2. corrected nodal finite-volume half-cell boundary treatment;
  3. Panel A passes through the Fig. 2E-F condition at I_total = 120 uM*um;
  4. Panel B passes through the Fig. 2E-F condition at k_PDE = 1.0 s^-1;
  5. readout is named distal relay exposure E_relay = integral D_relay dt.

Outputs:
  FigS3_pulse_fractionation_revised.pdf
  FigS3_pulse_fractionation_revised.png
  FigS3_metrics.csv
"""

from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from joblib import Parallel, delayed

OUTDIR = Path(__file__).resolve().parent / 'outputs'
OUTDIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.size': 8,
    'axes.titlesize': 9,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

BASE = dict(
    D0=50.0,          # um^2/s
    K_B=0.1,         # uM
    K_A=1.0,         # uM
    L=20.0,          # um
    N=150,
    distal_x=20.0,   # um
    resting_cAMP=0.005,  # uM = 5 nM
    t_end=12.0,      # s
)

PULSE_NUMBERS = np.array([1, 2, 4, 8, 16, 32, 64], dtype=int)
JIN_FIXED = 60.0      # uM*um/s
T_FIRST = 0.2         # s
T_LAST = 10.0         # s
MIN_TAU = 0.02        # s
N_JOBS = 1

# Panel A: fixed total input, vary effective PDE-mediated clearance
I_FIXED = 120.0
PDE_TO_PLOT = [0.50, 0.75, 1.00, 1.25, 1.50]

# Panel B: fixed clearance, vary total input
KPDE_FIXED = 1.00
INPUTS_TO_PLOT = [40.0, 80.0, 120.0, 160.0, 200.0]


def D_relay(c, P=BASE):
    c = np.clip(c, 1e-12, 1e4)
    return P['D0'] * (c / (P['K_B'] + c)) * (P['K_A'] / (P['K_A'] + c))


def make_pulse_schedule(I_total, Np, Jin=JIN_FIXED):
    """Equal integrated input with fixed instantaneous flux and uniformly spaced starts."""
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


def run_one_protocol(I_total, k_PDE, Np, P=BASE):
    starts, ends, tau = make_pulse_schedule(I_total, Np)
    if starts is None:
        return dict(ok=False, E_relay=np.nan, tau=tau, peak_c_dist=np.nan)

    x = np.linspace(0.0, P['L'], P['N'])
    dx = x[1] - x[0]
    distal_i = int(np.argmin(np.abs(x - P['distal_x'])))

    def rhs(t, C):
        C_eval = np.clip(C, 1e-12, 1e4)
        D_left = D_relay(C_eval[:-1], P)
        D_right = D_relay(C_eval[1:], P)
        # Harmonic interface averaging, as in revised Fig. 2E-F.
        D_mid = 2.0 * D_left * D_right / (D_left + D_right + 1e-30)
        flux = D_mid * (C_eval[1:] - C_eval[:-1]) / dx

        dC = np.zeros_like(C_eval)
        dC[1:-1] = (flux[1:] - flux[:-1]) / dx

        src = 1.0 if source_on(t, starts, ends) else 0.0

        # Correct nodal finite-volume half-cell boundary treatment.
        dC[0] = 2.0 * flux[0] / dx + 2.0 * JIN_FIXED * src / dx
        dC[-1] = -2.0 * flux[-1] / dx

        # Effective first-order PDE-mediated cAMP clearance.
        dC -= k_PDE * C_eval
        return dC

    C0 = np.full(P['N'], P['resting_cAMP'], dtype=float)
    t_eval = np.linspace(0.0, P['t_end'], 700)
    sol = solve_ivp(
        rhs,
        [0.0, P['t_end']],
        C0,
        t_eval=t_eval,
        method='BDF',
        rtol=1e-5,
        atol=1e-8,
        max_step=0.03,
    )

    if (not sol.success) or np.any(~np.isfinite(sol.y)):
        return dict(ok=False, E_relay=np.nan, tau=tau, peak_c_dist=np.nan)

    Cdist = sol.y[distal_i, :]
    Ddist = D_relay(Cdist, P)
    E_relay = np.trapezoid(Ddist, sol.t)

    return dict(
        ok=True,
        E_relay=float(E_relay),
        tau=float(tau),
        peak_c_dist=float(np.max(Cdist)),
    )


def run_key(kPDE, Itot, Np):
    return float(kPDE), float(Itot), int(Np), run_one_protocol(Itot, kPDE, Np)


# Run only conditions required for the two panels.
jobs = []
for k in PDE_TO_PLOT:
    for n in PULSE_NUMBERS:
        jobs.append((k, I_FIXED, n))
for I in INPUTS_TO_PLOT:
    for n in PULSE_NUMBERS:
        jobs.append((KPDE_FIXED, I, n))
jobs = sorted(set(jobs))

print(f'Running {len(jobs)} unique pulse-fractionation simulations...')
results_list = Parallel(n_jobs=N_JOBS, backend='loky', verbose=5)(
    delayed(run_key)(k, I, n) for k, I, n in jobs
)
results = {(k, I, n): out for k, I, n, out in results_list}

# Save numerical outputs for reproducibility.
metrics_path = OUTDIR / 'FigS3_metrics.csv'
with metrics_path.open('w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['k_PDE_s-1', 'I_total_uM_um', 'pulse_number', 'pulse_duration_s',
                     'E_relay_um2', 'peak_distal_cAMP_uM', 'ok'])
    for k, I, n in sorted(results):
        out = results[(k, I, n)]
        writer.writerow([
            k, I, n, out['tau'], out['E_relay'], out['peak_c_dist'], int(out['ok'])
        ])

# Make figure.
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))

ax = axes[0]
for kPDE in PDE_TO_PLOT:
    ns, vals = [], []
    for Np in PULSE_NUMBERS:
        out = results[(float(kPDE), float(I_FIXED), int(Np))]
        if out['ok'] and np.isfinite(out['E_relay']):
            ns.append(Np)
            vals.append(out['E_relay'])
    ax.plot(ns, vals, marker='o', lw=1.8,
            label=rf'$k_{{\rm PDE}}={kPDE:g}$ s$^{{-1}}$')
ax.set_xscale('log', base=2)
ax.set_xticks(PULSE_NUMBERS)
ax.set_xticklabels([str(int(n)) for n in PULSE_NUMBERS])
ax.minorticks_off()
ax.set_xlabel('Pulse number, N')
ax.set_ylabel(r'Distal relay exposure, $E_{\rm relay}$ ($\mu$m$^2$)')
ax.set_title(rf'A. Fixed input, $I_{{\rm total}}={I_FIXED:.0f}$ $\mu$M$\cdot\mu$m',
             fontweight='bold', loc='left')
ax.grid(alpha=0.25)
ax.legend(frameon=True, loc='best')

ax = axes[1]
for Itot in INPUTS_TO_PLOT:
    ns, vals = [], []
    for Np in PULSE_NUMBERS:
        out = results[(float(KPDE_FIXED), float(Itot), int(Np))]
        if out['ok'] and np.isfinite(out['E_relay']):
            ns.append(Np)
            vals.append(out['E_relay'])
    ax.plot(ns, vals, marker='o', lw=1.8,
            label=rf'$I_{{\rm total}}={Itot:.0f}$')
ax.set_xscale('log', base=2)
ax.set_xticks(PULSE_NUMBERS)
ax.set_xticklabels([str(int(n)) for n in PULSE_NUMBERS])
ax.minorticks_off()
ax.set_xlabel('Pulse number, N')
ax.set_ylabel(r'Distal relay exposure, $E_{\rm relay}$ ($\mu$m$^2$)')
ax.set_title(rf'B. Fixed clearance, $k_{{\rm PDE}}={KPDE_FIXED:g}$ s$^{{-1}}$',
             fontweight='bold', loc='left')
ax.grid(alpha=0.25)
ax.legend(frameon=True, loc='best')

fig.tight_layout()

pdf_path = OUTDIR / 'FigS3_pulse_fractionation_revised.pdf'
png_path = OUTDIR / 'FigS3_pulse_fractionation_revised.png'
fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
fig.savefig(png_path, dpi=300, bbox_inches='tight')
plt.close(fig)

# Print the Fig. 2E-F reference row so it is easy to verify exact agreement.
print('\nReference condition I_total=120, k_PDE=1.0 s^-1:')
for n in [1, 2, 4, 16]:
    out = results[(1.0, 120.0, n)]
    print(f'  N={n:2d}: E_relay={out["E_relay"]:.3f} um^2, tau={out["tau"]:.5f} s')

print(f'\nSaved {pdf_path}')
print(f'Saved {png_path}')
print(f'Saved {metrics_path}')
