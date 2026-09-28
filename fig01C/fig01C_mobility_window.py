from pathlib import Path
import csv
import json
import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

P = {
    "D_relay_0_um2_s": 50.0,
    "baseline_KB_uM": 0.1,
    "baseline_KA_uM": 1.0,
    "equal_KB_uM": 0.1,
    "equal_KA_uM": 0.1,
    "c_min_uM": 1e-3,
    "c_max_uM": 100.0,
    "n_points": 1000,
}


def d_relay(c, KB, KA, D0):
    return D0 * (c / (KB + c)) * (KA / (KA + c))


def analytic_peak(KB, KA, D0):
    c_peak = np.sqrt(KA * KB)
    return c_peak, d_relay(c_peak, KB, KA, D0)


c = np.logspace(np.log10(P["c_min_uM"]), np.log10(P["c_max_uM"]), P["n_points"])
D0 = P["D_relay_0_um2_s"]

KB1, KA1 = P["baseline_KB_uM"], P["baseline_KA_uM"]
KB2, KA2 = P["equal_KB_uM"], P["equal_KA_uM"]
y1 = d_relay(c, KB1, KA1, D0)
y2 = d_relay(c, KB2, KA2, D0)
cp1, dp1 = analytic_peak(KB1, KA1, D0)
cp2, dp2 = analytic_peak(KB2, KA2, D0)

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

fig, ax = plt.subplots(figsize=(5.2, 3.5))
ax.plot(c, y1, lw=2.2, label="Independent thresholds")
ax.plot(c, y2, lw=2.0, ls="--", label="Equal-threshold case")
ax.axvline(KB1, lw=1.0, ls=":", alpha=0.75)
ax.axvline(KA1, lw=1.0, ls=":", alpha=0.75)
ax.text(KB1 * 0.82, 21.3, r"$K_B$", ha="right", va="center")
ax.text(KA1 * 1.10, 21.3, r"$K_A$", ha="left", va="center")
ax.set_xscale("log")
ax.set_xlim(P["c_min_uM"], P["c_max_uM"])
ax.set_ylim(0, 32)
ax.set_xlabel(r"Local cAMP concentration ($\mu$M)")
ax.set_ylabel(r"$D_{\mathrm{relay}}$ ($\mu$m$^2$ s$^{-1}$)")
ax.set_title("Concentration-dependent mobility window", fontweight="bold")
ax.grid(alpha=0.22)
ax.legend(frameon=True, loc="upper right")
fig.tight_layout()

png = OUT / "fig01C_mobility_window.png"
pdf = OUT / "fig01C_mobility_window.pdf"
fig.savefig(png, dpi=300, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")
plt.close(fig)

with (OUT / "fig01C_metrics.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["condition", "K_B_uM", "K_A_uM", "c_peak_uM", "D_relay_peak_um2_s"])
    w.writerow(["baseline_independent_thresholds", KB1, KA1, cp1, dp1])
    w.writerow(["equal_thresholds", KB2, KA2, cp2, dp2])

with (OUT / "fig01C_parameters.json").open("w", encoding="utf-8") as f:
    json.dump(P, f, indent=2)

print(png)
print(pdf)
print(f"baseline c_peak={cp1:.6f} uM, D_peak={dp1:.6f} um2/s")
print(f"equal c_peak={cp2:.6f} uM, D_peak={dp2:.6f} um2/s")
