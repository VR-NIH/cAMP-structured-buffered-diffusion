import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / 'outputs'
OUT.mkdir(parents=True, exist_ok=True)

# Binding thresholds shared by the organized and conventional limits.
K_B = 0.1   # uM
K_A = 1.0   # uM
K_D_EFF = np.sqrt(K_A * K_B)  # uM

# Plot grids.
c = np.logspace(-2, 1, 320)          # uM
rho_R = np.logspace(-2, 2, 320)      # R_T / R_* (dimensionless)
beta = np.logspace(-2, 2, 320)       # B_T / K_D,eff (dimensionless)

C1, RHO = np.meshgrid(c, rho_R)
C2, BETA = np.meshgrid(c, beta)

# Organized RI: occupancy dependence times an explicitly illustrative
# architecture factor. No absolute R_T optimum is asserted.
def occupancy(c):
    return (c / (K_B + c)) * (K_A / (K_A + c))

def g_arch(rho):
    return 4.0 * rho / (1.0 + rho)**2

relay_raw = occupancy(C1) * g_arch(RHO)
relay_norm = relay_raw / np.nanmax(relay_raw)

# Conventional buffered diffusion: rapid-equilibrium buffer-capacity limit.
# Let beta = B_T / K_D,eff and u = c / K_D,eff.
# Then kappa = beta / (1 + u)^2 and D_app/D_free = 1/(1+kappa).
u = C2 / K_D_EFF
kappa = BETA / (1.0 + u)**2
buffer_ratio = 1.0 / (1.0 + kappa)

c_max = np.sqrt(K_A * K_B)
rho_max = 1.0

plt.rcParams.update({
    'font.size': 9,
    'axes.titlesize': 9.5,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

fig, axes = plt.subplots(1, 2, figsize=(8.25, 3.35))
fig.subplots_adjust(left=0.10, right=0.94, bottom=0.17, top=0.86, wspace=0.62)
levels = np.linspace(0, 1, 101)
contours = [0.25, 0.50, 0.75]

# A. Organized RI
ax = axes[0]
cf1 = ax.contourf(C1, RHO, relay_norm, levels=levels, cmap='viridis')
cs1 = ax.contour(C1, RHO, relay_norm, levels=contours, colors='white', linewidths=0.9)
ax.clabel(cs1, inline=True, fontsize=7, fmt='%.2f')
ax.plot(c_max, rho_max, marker='o', ms=5.2, mfc='none', mec='white', mew=1.4)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(c.min(), c.max())
ax.set_ylim(rho_R.min(), rho_R.max())
ax.set_xlabel(r'Local cAMP, $c$ ($\mu$M)')
ax.set_ylabel('Relative organized RI abundance\n' + r'$\rho_R=R_T/R_*$', labelpad=5)
ax.set_title('A. Organized RI: structured buffered diffusion', loc='center', fontweight='bold', pad=7)
cbar1 = fig.colorbar(cf1, ax=ax, pad=0.035, fraction=0.050)
cbar1.set_label(r'Normalized $D_{\mathrm{relay}}$')

# B. Unorganized RI
ax = axes[1]
cf2 = ax.contourf(C2, BETA, buffer_ratio, levels=levels, cmap='viridis')
cs2 = ax.contour(C2, BETA, buffer_ratio, levels=contours, colors='white', linewidths=0.9)
ax.clabel(cs2, inline=True, fontsize=7, fmt='%.2f')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(c.min(), c.max())
ax.set_ylim(beta.min(), beta.max())
ax.set_xlabel(r'Local cAMP, $c$ ($\mu$M)')
ax.set_ylabel('Dimensionless buffer abundance\n' + r'$\beta=B_T/K_{D,\mathrm{eff}}$', labelpad=5)
ax.set_title('B. Unorganized RI: conventional buffered diffusion', loc='center', fontweight='bold', pad=7)
cbar2 = fig.colorbar(cf2, ax=ax, pad=0.035, fraction=0.050)
cbar2.set_label(r'$D_{\mathrm{buffer}}/D_{\mathrm{free}}$')

fig.savefig(OUT / 'fig06.pdf', dpi=300, bbox_inches='tight')
fig.savefig(OUT / 'fig06.png', dpi=300, bbox_inches='tight')
plt.close(fig)

params = {
    'K_B_uM': K_B,
    'K_A_uM': K_A,
    'K_D_eff_uM': float(K_D_EFF),
    'cAMP_range_uM': [float(c.min()), float(c.max())],
    'rho_R_range': [float(rho_R.min()), float(rho_R.max())],
    'beta_range': [float(beta.min()), float(beta.max())],
    'organized_architecture_factor': 'g_R(rho)=4*rho/(1+rho)^2',
    'organized_surface': 'Drelay_norm = normalize[g_R(rho_R) * c/(K_B+c) * K_A/(K_A+c)]',
    'conventional_surface': 'kappa=beta/(1+c/K_D_eff)^2; Dbuffer/Dfree=1/(1+kappa)',
    'organized_surface_maximum': {'c_uM': float(c_max), 'rho_R': rho_max},
    'interpretation': {
        'rho_R': 'R_T/R_*; R_* is an illustrative reference abundance and is not inferred from Fig. 5',
        'beta': 'B_T/K_D_eff; dimensionless buffer abundance in the conventional rapid-equilibrium limit',
        'colorbars': 'separate scales; panels are limiting regimes and are not added pointwise'
    }
}
with open(OUT / 'fig06_parameters.json', 'w') as f:
    json.dump(params, f, indent=2)

with open(OUT / 'fig06_parameters.txt', 'w') as f:
    f.write('FIGURE 6 REVISED PARAMETERS AND DEFINITIONS\n')
    f.write('==========================================\n\n')
    f.write('Shared cAMP-binding thresholds\n')
    f.write(f'K_B = {K_B:.3g} uM\n')
    f.write(f'K_A = {K_A:.3g} uM\n')
    f.write(f'K_D,eff = sqrt(K_A*K_B) = {K_D_EFF:.6f} uM\n')
    f.write(f'c_max = sqrt(K_A*K_B) = {c_max:.6f} uM\n\n')
    f.write('Panel A: organized RI / structured buffered diffusion\n')
    f.write('rho_R = R_T / R_* (dimensionless)\n')
    f.write('R_* is an illustrative reference abundance; no absolute RI concentration optimum is asserted.\n')
    f.write('g_R(rho_R) = 4*rho_R/(1+rho_R)^2\n')
    f.write('D_relay(c,rho_R) proportional to g_R(rho_R) * [c/(K_B+c)] * [K_A/(K_A+c)]\n')
    f.write('Displayed surface = D_relay normalized to its own maximum.\n')
    f.write('Surface maximum: rho_R = 1, c = sqrt(K_A*K_B).\n')
    f.write('rho_R plotted from 0.01 to 100.\n\n')
    f.write('Panel B: unorganized RI / conventional buffered diffusion\n')
    f.write('K_D,eff = sqrt(K_A*K_B)\n')
    f.write('beta = B_T / K_D,eff (dimensionless buffer abundance)\n')
    f.write('u = c / K_D,eff\n')
    f.write('kappa = beta/(1+u)^2\n')
    f.write('D_buffer/D_free = 1/(1+kappa)\n')
    f.write('beta plotted from 0.01 to 100.\n\n')
    f.write('Interpretive constraints\n')
    f.write('- Panel A cAMP dependence is derived from the source/acceptor occupancy closure.\n')
    f.write('- Panel A RI-abundance dependence is explicitly phenomenological and illustrative.\n')
    f.write('- R_* and g_R are not inferred from the Fig. 5 R_T optimum.\n')
    f.write('- Panel B is the conventional rapid-equilibrium buffering limit.\n')
    f.write('- The panels are limiting effective transport regimes, not parallel diffusion coefficients to be added pointwise.\n')
    f.write('- Separate colorbars are used because normalized D_relay and D_buffer/D_free are different quantities.\n')

# Small numerical summary for reproducibility.
with open(OUT / 'fig06_summary.txt', 'w') as f:
    f.write(f'c_max_uM\t{c_max:.8f}\n')
    f.write(f'rho_R_max\t{rho_max:.8f}\n')
    f.write(f'K_D_eff_uM\t{K_D_EFF:.8f}\n')
