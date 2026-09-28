from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Fig S2: Sensitivity of relay-mobility window to KA and KB thresholds
# All concentrations are in uM; D0 in um^2/s.

D0 = 50.0
c = np.logspace(-3, 1.5, 600)  # 1 nM to 31.6 uM for plotting

# Threshold pairs spanning 10 nM to 1 uM.
# Pairs are chosen to test shifts in the transport window without duplicating
# separate KA-only and KB-only scans in the main figure.
pairs = [
    (0.01, 0.10),
    (0.03, 0.30),
    (0.10, 1.00),
    (0.30, 1.00),
    (0.01, 1.00),
    (0.10, 0.10),
]

# Label strings avoid unicode for compatibility in source.
labels = [
    r"$K_B=0.01$, $K_A=0.10$",
    r"$K_B=0.03$, $K_A=0.30$",
    r"$K_B=0.10$, $K_A=1.00$",
    r"$K_B=0.30$, $K_A=1.00$",
    r"$K_B=0.01$, $K_A=1.00$",
    r"$K_B=K_A=0.10$",
]


def d_relay(c, KB, KA):
    return D0 * (c / (KB + c)) * (KA / (KA + c))

# Heatmap grid for cmax = sqrt(KA KB)
Kvals = np.logspace(-2, 0, 200)  # 10 nM to 1 uM
KB_grid, KA_grid = np.meshgrid(Kvals, Kvals)
cmax_grid = np.sqrt(KA_grid * KB_grid)

plt.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.45))

# Panel A: normalized curves
ax = axes[0]
for (KB, KA), lab in zip(pairs, labels):
    y = d_relay(c, KB, KA)
    y_norm = y / np.max(y)
    ax.plot(c, y_norm, lw=1.8, label=lab)
    cm = np.sqrt(KB * KA)
    ym = d_relay(cm, KB, KA) / np.max(y)
    ax.plot(cm, ym, marker='o', ms=3)

ax.set_xscale('log')
ax.set_xlim(1e-3, 10)
ax.set_ylim(0, 1.06)
ax.set_xlabel(r"Local cAMP, $c$ ($\mu$M)")
ax.set_ylabel(r"Normalized $D_{\mathrm{relay}}(c)$")
ax.set_title("A. Threshold variation preserves the window", loc='left', fontweight='bold')
ax.grid(True, which='major', alpha=0.22)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.27), framealpha=0.95, ncol=2, title=r'Threshold pairs ($\mu$M)', title_fontsize=6.5, borderaxespad=0.0)

# Panel B: cmax heatmap
ax = axes[1]
im = ax.pcolormesh(KB_grid, KA_grid, cmax_grid, shading='auto', norm=LogNorm(vmin=1e-2, vmax=1), cmap='viridis')
levels = [0.03, 0.1, 0.316, 1.0]
cs = ax.contour(KB_grid, KA_grid, cmax_grid, levels=levels, colors='white', linewidths=0.9)
ax.clabel(cs, inline=True, fontsize=6.5, fmt=lambda v: f"{v:g}")
ax.scatter([0.1], [1.0], s=36, marker='*', color='black', zorder=5, label='reference')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r"Source-loading threshold, $K_B$ ($\mu$M)")
ax.set_ylabel(r"Acceptor-vacancy threshold, $K_A$ ($\mu$M)")
ax.set_title(r"B. Window position follows $c_{\max}=\sqrt{K_AK_B}$", loc='left', fontweight='bold')
ax.legend(loc='upper left', framealpha=0.95, fontsize=6.5)
cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
cb.set_label(r"$c_{\max}$ ($\mu$M)")

fig.subplots_adjust(left=0.08, right=0.98, bottom=0.30, top=0.91, wspace=0.36)

outdir = Path(__file__).resolve().parent / "outputs"
outdir.mkdir(parents=True, exist_ok=True)
out_png = outdir / "FigS2_threshold_sensitivity.png"
out_pdf = outdir / "FigS2_threshold_sensitivity.pdf"
fig.savefig(out_png, dpi=300, bbox_inches='tight')
fig.savefig(out_pdf, dpi=300, bbox_inches='tight')
plt.close(fig)

# Write parameter summary
summary = f"""Fig S2 parameter summary: threshold sensitivity of relay-mobility window

Purpose
- Test whether the relay-mobility window depends on the reference choice of K_A and K_B.
- Demonstrate that the window persists when effective occupancy thresholds span 10 nM to 1 uM.

Model
D_relay(c) = D0 * [c/(K_B+c)] * [K_A/(K_A+c)]
D0 = {D0} um^2/s
cmax = sqrt(K_A*K_B)
Curves in panel A are normalized to their own maximum.

Concentration axis
c range = 0.001 to 31.6 uM for computation; plotted to 10 uM.

Threshold pairs in panel A, in uM
"""
for KB, KA in pairs:
    cm = np.sqrt(KB * KA)
    peak = d_relay(cm, KB, KA)
    summary += f"- K_B={KB:g}, K_A={KA:g}; cmax={cm:g} uM; peak D_relay={peak:g} um^2/s\n"
summary += """
Panel B
K_B and K_A each varied from 0.01 to 1.0 uM on a logarithmic grid.
Heatmap shows cmax = sqrt(K_A*K_B).
White contours show cmax = 0.03, 0.1, 0.316, and 1.0 uM.
Black star marks the reference pair K_B=0.1 uM, K_A=1.0 uM.

Interpretation
Changing K_A or K_B shifts the concentration range of maximal relay-associated mobility,
but does not remove the non-monotonic source-acceptor mobility window. The existence of the
window follows from multiplying a rising source-availability term by a falling acceptor-vacancy term.
"""
with open(outdir / "FigS2_parameter_summary.txt", "w") as f:
    f.write(summary)

print(out_png)
print(out_pdf)
