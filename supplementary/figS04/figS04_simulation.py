from pathlib import Path
import csv
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Figure S4: sensitivity to PDE sink form.
# This corrected version inherits the final Fig. 2 numerical implementation.

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

HERE = Path(__file__).resolve().parent
OUT = HERE / 'outputs'
OUT.mkdir(parents=True, exist_ok=True)

# Shared structured-buffered mobility parameters (match final Fig. 2).
KB = 0.1
KA = 1.0
D0 = 50.0
CMAX = float(np.sqrt(KB * KA))

# Final Fig. 2 solver settings.
METHOD = 'BDF'
RTOL = 1e-5
ATOL = 1e-8
MAX_STEP = 0.03


def D_relay(c):
    v = np.clip(c, 1e-12, 1e4)
    return D0 * (v / (KB + v)) * (KA / (KA + v))


def pde_sink(c, mode, k_ref, KM=None):
    """Clearance term. MM is matched to first-order at c = CMAX."""
    c = np.clip(c, 0.0, 1e4)
    if mode == 'linear':
        return k_ref * c
    if mode == 'mm':
        if KM is None:
            raise ValueError('KM is required for Michaelis-Menten mode')
        Vmax = k_ref * (KM + CMAX)
        return Vmax * c / (KM + c + 1e-30)
    raise ValueError(mode)


# -----------------------------------------------------------------------------
# Panels A-C: exact final Fig. 2 A-D single-pulse transport implementation,
# changing only the PDE sink law.
# -----------------------------------------------------------------------------
L1 = 50.0
N1 = 150
s1 = np.linspace(0.0, L1, N1)
ds1 = s1[1] - s1[0]
SRC_CONC = 2.5
SRC_OFF = 1.0
T_SINGLE = 8.0
N_EVAL_SINGLE = 600
T_PLOT = 5.0
K_PDE_SINGLE = 0.1
PEN_FRAC = 0.1


def make_single_rhs(mode, KM, proximal_boundary):
    if proximal_boundary not in {'clamped', 'reflecting'}:
        raise ValueError(proximal_boundary)

    def rhs(t, C):
        C = np.clip(C, 0.0, 1e4)
        dC = np.zeros_like(C)

        # Arithmetic interface averaging, as in final Fig. 2 A-D.
        D_mid = 0.5 * (D_relay(C[:-1]) + D_relay(C[1:]))
        flux = D_mid * (C[1:] - C[:-1]) / ds1
        dC[1:-1] = (flux[1:] - flux[:-1]) / ds1

        # Half-width boundary control volumes.
        if proximal_boundary == 'reflecting':
            dC[0] = 2.0 * flux[0] / ds1
        else:
            dC[0] = 0.0
        dC[-1] = -2.0 * flux[-1] / ds1

        dC -= pde_sink(C, mode, K_PDE_SINGLE, KM)

        # Keep the proximal node clamped during the 1-s input despite sink.
        if proximal_boundary == 'clamped':
            dC[0] = 0.0
        return dC

    return rhs


def run_single(mode='linear', KM=None):
    rhs_clamped = make_single_rhs(mode, KM, 'clamped')
    rhs_reflecting = make_single_rhs(mode, KM, 'reflecting')
    t_eval = np.linspace(0.0, T_SINGLE, N_EVAL_SINGLE)

    C0 = np.zeros(N1)
    C0[0] = SRC_CONC

    t_ev1 = t_eval[t_eval <= SRC_OFF]
    if len(t_ev1) == 0 or t_ev1[-1] < SRC_OFF:
        t_ev1 = np.append(t_ev1, SRC_OFF)
    sol1 = solve_ivp(
        rhs_clamped, [0.0, SRC_OFF], C0,
        t_eval=t_ev1, method=METHOD, rtol=RTOL, atol=ATOL, max_step=MAX_STEP,
    )
    if not sol1.success:
        raise RuntimeError(sol1.message)
    sol1.y[0, :] = SRC_CONC

    # Release concentration clamp into a reflecting proximal boundary.
    C1 = np.clip(sol1.y[:, -1], 0.0, 1e4)
    t_ev2 = t_eval[t_eval >= SRC_OFF]
    if len(t_ev2) == 0 or t_ev2[0] > SRC_OFF:
        t_ev2 = np.insert(t_ev2, 0, SRC_OFF)
    sol2 = solve_ivp(
        rhs_reflecting, [SRC_OFF, T_SINGLE], C1,
        t_eval=t_ev2, method=METHOD, rtol=RTOL, atol=ATOL, max_step=MAX_STEP,
    )
    if not sol2.success:
        raise RuntimeError(sol2.message)

    start2 = 1 if (len(sol2.t) and np.isclose(sol2.t[0], sol1.t[-1])) else 0
    t = np.hstack([sol1.t, sol2.t[start2:]])
    C = np.hstack([sol1.y, sol2.y[:, start2:]])
    return t, C


def penetration_depth(max_profile):
    """10%-of-peak penetration depth with linear interpolation.

    Returns NaN if the threshold is not reached within the domain.
    """
    peak = float(np.max(max_profile))
    threshold = PEN_FRAC * peak
    below = np.where(max_profile < threshold)[0]
    if len(below) == 0:
        return np.nan
    i = int(below[0])
    if i == 0:
        return float(s1[0])
    x0, x1 = float(s1[i - 1]), float(s1[i])
    y0, y1 = float(max_profile[i - 1]), float(max_profile[i])
    if np.isclose(y1, y0):
        return x1
    frac = (threshold - y0) / (y1 - y0)
    return x0 + frac * (x1 - x0)


# -----------------------------------------------------------------------------
# Panel D: exact final Fig. 2 E-F equal-input implementation,
# changing only the PDE sink law.
# -----------------------------------------------------------------------------
L2 = 20.0
N2 = 150
s2 = np.linspace(0.0, L2, N2)
ds2 = s2[1] - s2[0]
DISTAL_S = 20.0
DISTAL_I = int(np.argmin(np.abs(s2 - DISTAL_S)))
RESTING_CAMP = 0.005
T_PULSE = 12.0
N_EVAL_PULSE = 700
I_TOTAL = 120.0
K_PDE_PULSE = 1.0
PULSE_NUMBERS = [1, 2, 4, 16]
JIN_FIXED = 60.0
T_FIRST = 0.2
T_LAST = 10.0
MIN_TAU = 0.02


def make_pulse_schedule(I_total, Np, Jin=JIN_FIXED):
    tau = I_total / (Np * Jin)
    if tau < MIN_TAU:
        return None, None, np.nan
    interval = (T_LAST - T_FIRST) / Np
    if tau > interval:
        return None, None, np.nan
    starts = T_FIRST + np.arange(Np) * interval
    ends = starts + tau
    return starts, ends, tau


def source_on(t, starts, ends):
    return bool(np.any((t >= starts) & (t < ends)))


def run_pulse_protocol(Np, mode='linear', KM=None):
    starts, ends, tau = make_pulse_schedule(I_TOTAL, Np)
    if starts is None:
        raise ValueError(f'invalid pulse schedule N={Np}')

    def rhs(t, C):
        C = np.clip(C, 1e-12, 1e4)
        Dl = D_relay(C[:-1])
        Dr = D_relay(C[1:])
        # Harmonic interface averaging, as in final Fig. 2 E-F.
        Dm = 2.0 * Dl * Dr / (Dl + Dr + 1e-30)
        flux = Dm * (C[1:] - C[:-1]) / ds2

        dC = np.zeros_like(C)
        dC[1:-1] = (flux[1:] - flux[:-1]) / ds2
        src = 1.0 if source_on(t, starts, ends) else 0.0

        # Correct half-width boundary control volumes.
        dC[0] = 2.0 * flux[0] / ds2 + 2.0 * JIN_FIXED * src / ds2
        dC[-1] = -2.0 * flux[-1] / ds2

        dC -= pde_sink(C, mode, K_PDE_PULSE, KM)
        return dC

    C0 = np.full(N2, RESTING_CAMP)
    t_eval = np.linspace(0.0, T_PULSE, N_EVAL_PULSE)
    sol = solve_ivp(
        rhs, [0.0, T_PULSE], C0,
        t_eval=t_eval, method=METHOD, rtol=RTOL, atol=ATOL, max_step=MAX_STEP,
    )
    if not sol.success or np.any(~np.isfinite(sol.y)):
        raise RuntimeError(sol.message)

    Cdist = sol.y[DISTAL_I, :]
    Ddist = D_relay(Cdist)
    exposure = float(np.trapezoid(Ddist, sol.t))
    return dict(
        t=sol.t,
        Ddist=Ddist,
        Cdist=Cdist,
        exposure=exposure,
        starts=starts,
        ends=ends,
        tau=float(tau),
    )


KM_VALUES = [0.1, 0.3, 1.0]
CONDITIONS = [('First-order', 'linear', None)] + [
    (rf'MM $K_M$={KM:g} $\mu$M', 'mm', KM) for KM in KM_VALUES
]

print('Running corrected Fig. S4 single-pulse comparisons...')
single_results = []
for label, mode, KM in CONDITIONS:
    print(' single:', label)
    t, C = run_single(mode, KM)
    max_profile = C.max(axis=1)
    depth = penetration_depth(max_profile)
    single_results.append(dict(
        label=label, mode=mode, KM=KM, t=t, C=C,
        max_profile=max_profile, depth=depth,
    ))

print('Running corrected Fig. S4 equal-input comparisons...')
pulse_results = []
for label, mode, KM in CONDITIONS:
    for Np in PULSE_NUMBERS:
        print(' pulse:', label, 'N=', Np)
        out = run_pulse_protocol(Np, mode, KM)
        pulse_results.append(dict(label=label, mode=mode, KM=KM, Np=Np, **out))

# Save quantitative outputs.
rows = []
for r in single_results:
    rows.append(dict(
        kind='single', label=r['label'], mode=r['mode'], KM_uM=r['KM'],
        penetration_depth_um=r['depth'], pulse_number=np.nan,
        relay_exposure_um2=np.nan, pulse_duration_s=np.nan,
    ))
for r in pulse_results:
    rows.append(dict(
        kind='pulse', label=r['label'], mode=r['mode'], KM_uM=r['KM'],
        penetration_depth_um=np.nan, pulse_number=r['Np'],
        relay_exposure_um2=r['exposure'], pulse_duration_s=r['tau'],
    ))
metrics = pd.DataFrame(rows)
metrics.to_csv(OUT / 'FigS4_metrics.csv', index=False)

# Save compact parameters.
params = {
    'shared': {'D0_um2_s': D0, 'K_B_uM': KB, 'K_A_uM': KA, 'cmax_uM': CMAX},
    'MM_matching': 'Vmax = k_ref * (KM + cmax), so v_MM(cmax) = k_ref*cmax',
    'KM_values_uM': KM_VALUES,
    'single_pulse': {
        'L_um': L1, 'N': N1, 'source_concentration_uM': SRC_CONC,
        'source_duration_s': SRC_OFF, 't_end_s': T_SINGLE,
        'k_PDE_reference_s-1': K_PDE_SINGLE,
        'interface_averaging': 'arithmetic',
        'post_pulse_proximal_boundary': 'reflecting half-cell',
        'distal_boundary': 'reflecting half-cell',
        'penetration_threshold_fraction': PEN_FRAC,
    },
    'pulse_fractionation': {
        'L_um': L2, 'N': N2, 'distal_s_um': DISTAL_S,
        'resting_cAMP_uM': RESTING_CAMP, 't_end_s': T_PULSE,
        'I_total_uM_um': I_TOTAL, 'J_in_uM_um_s': JIN_FIXED,
        'k_PDE_reference_s-1': K_PDE_PULSE,
        'pulse_numbers': PULSE_NUMBERS,
        'interface_averaging': 'harmonic',
        'proximal_boundary': 'imposed flux half-cell',
        'distal_boundary': 'reflecting half-cell',
    },
    'solver': {'method': METHOD, 'rtol': RTOL, 'atol': ATOL, 'max_step_s': MAX_STEP},
}
with (OUT / 'FigS4_parameters.json').open('w') as fh:
    json.dump(params, fh, indent=2)

# Plot.
fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.2))
axA, axB, axC, axD = axes.flat

# A: max cAMP profile
for r in single_results:
    ls = '-' if r['mode'] == 'linear' else '--'
    axA.plot(s1, r['max_profile'], lw=1.8, ls=ls, label=r['label'])
axA.axhline(CMAX, color='0.55', ls=':', lw=1.0, label=r'$c_{\max}$')
axA.set_xlim(0, 25)
axA.set_xlabel(r'RI-network coordinate, $s$ ($\mu$m)')
axA.set_ylabel(r'Max free cAMP ($\mu$M)')
axA.set_title('A. Single-pulse propagation', loc='left', fontweight='bold')
axA.grid(alpha=0.25)
axA.legend(frameon=True, fontsize=6.2)

# B: local mobility at 10 um
S_PROBE = 10.0
li = int(np.argmin(np.abs(s1 - S_PROBE)))
for r in single_results:
    ls = '-' if r['mode'] == 'linear' else '--'
    axB.plot(r['t'], D_relay(r['C'][li, :]), lw=1.8, ls=ls, label=r['label'])
axB.set_xlim(0, T_PLOT)
axB.set_xlabel('Time (s)')
axB.set_ylabel(r'$D_{\rm relay}[c(s=10\,\mu m,t)]$ ($\mu$m$^2$ s$^{-1}$)')
axB.set_title('B. Local mobility window is preserved', loc='left', fontweight='bold')
axB.grid(alpha=0.25)

# C: penetration depth
labels_short = ['First-order'] + [f'MM\n$K_M$={KM:g}' for KM in KM_VALUES]
depths = [r['depth'] for r in single_results]
axC.bar(np.arange(len(depths)), depths)
axC.set_xticks(np.arange(len(depths)))
axC.set_xticklabels(labels_short)
axC.set_ylabel(r'Penetration depth along $s$ ($\mu$m)')
axC.set_title('C. Quantitative range sensitivity', loc='left', fontweight='bold')
axC.grid(axis='y', alpha=0.25)

# D: equal-input distal mobility exposure
for label, mode, KM in CONDITIONS:
    subset = [r for r in pulse_results if r['label'] == label]
    xs = [r['Np'] for r in subset]
    ys = [r['exposure'] for r in subset]
    ls = '-' if mode == 'linear' else '--'
    axD.plot(xs, ys, marker='o', lw=1.8, ls=ls, label=label)
axD.set_xscale('log', base=2)
axD.set_xticks(PULSE_NUMBERS)
axD.set_xticklabels([str(n) for n in PULSE_NUMBERS])
axD.set_xlabel('Pulse number, N')
axD.set_ylabel(r'Distal mobility exposure, $E_{\rm relay}$ ($\mu$m$^2$)')
axD.set_title('D. Temporal packaging remains regime dependent', loc='left', fontweight='bold')
axD.grid(alpha=0.25)
axD.legend(frameon=True, fontsize=6.2)

fig.tight_layout()
png = OUT / 'FigS4_PDE_sensitivity_corrected.png'
pdf = OUT / 'FigS4_PDE_sensitivity_corrected.pdf'
fig.savefig(png, dpi=300, bbox_inches='tight')
fig.savefig(pdf, dpi=300, bbox_inches='tight')
plt.close(fig)

# Human-readable summary.
with (OUT / 'FigS4_summary.txt').open('w') as fh:
    fh.write('FIGURE S4 CORRECTED PDE-SINK SENSITIVITY\n')
    fh.write('========================================\n\n')
    fh.write('All transport/numerical settings inherit final Fig. 2; only PDE sink form changes.\n')
    fh.write(f'cmax = {CMAX:.9g} uM\n')
    fh.write('MM matching: Vmax = k_ref*(KM+cmax)\n\n')
    fh.write('Single-pulse penetration depths:\n')
    for r in single_results:
        fh.write(f"  {r['label']}: {r['depth']:.6g} um\n")
    fh.write('\nEqual-input distal mobility exposure E_relay (um^2):\n')
    for label, mode, KM in CONDITIONS:
        vals = [r for r in pulse_results if r['label'] == label]
        fh.write(f'  {label}: ' + ', '.join(f"N={r['Np']}: {r['exposure']:.6g}" for r in vals) + '\n')

print('\nCorrected Fig. S4 penetration depths:')
for r in single_results:
    print(f"  {r['label']}: {r['depth']:.6f} um")
print('\nCorrected Fig. S4 distal mobility exposure:')
for label, mode, KM in CONDITIONS:
    vals = [r for r in pulse_results if r['label'] == label]
    print(' ', label, ':', ', '.join(f"N={r['Np']} {r['exposure']:.6f}" for r in vals))
print('\nSaved:', png)
print('Saved:', pdf)
print('Saved:', OUT / 'FigS4_metrics.csv')
