import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

df = pd.read_csv("data/ising_fss_betac.csv")
L  = df["L"].values.astype(float)

# Instead, save each plot as its own figure:

FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

COLORS = ["#378ADD","#1D9E75","#D85A30","#BA7517","#D4537E"]
ERR_KW = dict(fmt="o", capsize=4, elinewidth=1.0, markersize=6)

# ── Plot 1 ─────────────────────────────────────────────────────────────────
fig, ax =  plt.subplots(figsize=(6, 4))
chi = df["susceptibility"].values
dchi = df["dsusceptibility"].values
ax.errorbar(np.log(L), np.log(chi), yerr=dchi/chi,
            color=COLORS[0], **ERR_KW, label="data")
slope, intercept = np.polyfit(np.log(L), np.log(chi), 1)
ax.plot(np.log(L), slope*np.log(L)+intercept, "--", color=COLORS[0],
        alpha=0.6, label=rf"slope = {slope:.3f}  ")# (expect 7/4 = 1.75)")
ax.set_xlabel(r"$\ln L$")
ax.set_ylabel(r"$\ln \chi$")
ax.set_title(r"Susceptibility $\chi$ vs $L$")
ax.set_xticks(np.log(L));  ax.set_xticklabels([str(int(l)) for l in L])
ax.legend(fontsize=8);  ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "fss_chi.pdf")
plt.close()

# ── Plot 2 ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
C  = df["specific_heat"].values
dC = df["dspecific_heat"].values
ax.errorbar(np.log(L), np.log(C), yerr=dC/C,
            color=COLORS[1], **ERR_KW, label="data")
slope_C, ic = np.polyfit(np.log(L), np.log(C), 1)
ax.plot(np.log(L), slope_C*np.log(L)+ic, "--", color=COLORS[1],
        alpha=0.6, label=rf"slope = {slope_C:.3f}  ")# (expect 0, log divergence)")
ax.set_xlabel(r"$\ln L$");  ax.set_ylabel(r"$\ln C$")
ax.set_title(r"Specific heat $C$ vs $L$")
ax.set_xticks(np.log(L));  ax.set_xticklabels([str(int(l)) for l in L])
ax.legend(fontsize=8);  ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "fss_specific_heat.pdf")
plt.close()

# ── Plot 3 ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
U4  = df["U4"].values
dU4 = df["dU4"].values
ax.errorbar(L, U4, yerr=dU4, color=COLORS[2], **ERR_KW, label=r"$U_4$")
# ax.axhline(0.6107, ls="--", color="gray", lw=0.9,
#            label=r"universal value $\approx 0.611$")
ax.set_xlabel(r"$L$");  ax.set_ylabel(r"$U_4$")
ax.set_title(r"Binder cumulant $U_4$ vs $L$")
ax.set_xscale("log", base=2)
ax.set_xticks(L);  ax.set_xticklabels([str(int(l)) for l in L])
ax.legend(fontsize=8);  ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "fss_U4.pdf")
plt.close()

# ── Plot 4 ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
U2  = df["U2"].values
dU2 = df["dU2"].values
ax.errorbar(L, U2, yerr=dU2, color=COLORS[3], **ERR_KW, label=r"$U_2$")
ax.set_xlabel(r"$L$");  ax.set_ylabel(r"$U_2$")
ax.set_title(r"Binder cumulant $U_2$ vs $L$")
ax.set_xscale("log", base=2)
ax.set_xticks(L);  ax.set_xticklabels([str(int(l)) for l in L])
ax.legend(fontsize=8);  ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "fss_U2.pdf")
plt.close()

# ── Plot 5 ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
dU4b  = df["dU4_dbeta"].values
ddU4b = df["ddU4_dbeta"].values
ax.errorbar(np.log(L), np.log(dU4b), yerr=ddU4b/dU4b,
            color=COLORS[4], **ERR_KW, label=r"$dU_4/d\beta$")
slope_dU4, ic_dU4 = np.polyfit(np.log(L), np.log(dU4b), 1)
ax.plot(np.log(L), slope_dU4*np.log(L)+ic_dU4, "--", color=COLORS[4],
        alpha=0.6, label=rf"slope = {slope_dU4:.3f}  ")# (expect 1/ν = 1.0)")
ax.set_xlabel(r"$\ln L$");  ax.set_ylabel(r"$\ln |dU_4/d\beta|$")
ax.set_title(r"$dU_4/d\beta$ vs $L$")
ax.set_xticks(np.log(L));  ax.set_xticklabels([str(int(l)) for l in L])
ax.legend(fontsize=8);  ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "fss_dU4_dbeta.pdf")
plt.close()

# ── Plot 6 ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
dlnm2  = df["dlnm2_dbeta"].values
ddlnm2 = df["ddlnm2_dbeta"].values
ax.errorbar(np.log(L), np.log(dlnm2), yerr=ddlnm2/dlnm2,
            color=COLORS[0], **ERR_KW, label=r"$d\ln\langle m^2\rangle/d\beta$")
slope_lnm2, ic_lnm2 = np.polyfit(np.log(L), np.log(dlnm2), 1)
ax.plot(np.log(L), slope_lnm2*np.log(L)+ic_lnm2, "--", color=COLORS[0],
        alpha=0.6, label=rf"slope = {slope_lnm2:.3f}  ")# (expect 1/ν = 1.0)")
ax.set_xlabel(r"$\ln L$");  ax.set_ylabel(r"$\ln |d\ln\langle m^2\rangle/d\beta|$")
ax.set_title(r"$d\ln\langle m^2\rangle/d\beta$ vs $L$")
ax.set_xticks(np.log(L));  ax.set_xticklabels([str(int(l)) for l in L])
ax.legend(fontsize=8);  ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "fss_dlnm2_dbeta.pdf")
plt.close()