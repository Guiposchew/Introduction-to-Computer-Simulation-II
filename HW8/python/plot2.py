from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

L = 128
N = L * L

# ── Load data ─────────────────────────────────────────────────────────────
df = pd.read_csv("HW8/python/data/hw8_problem16_single_L128_scan.csv")
beta = df["beta"].values

def parse_cluster_sizes(s):
    s = str(s).replace("\n", " ")
    return np.array(list(map(int, re.findall(r"-?\d+", s))))

df["cluster_sizes_array"] = df["cluster_sizes"].apply(parse_cluster_sizes)
cl_list = df["cluster_sizes_array"].values

beta_c = np.log(1 + np.sqrt(2)) / 2   # exact: ≈ 0.440687

# ── FIX: correct Wolff improved estimator ─────────────────────────────────
# WRONG:   chi = np.mean(cl**2) / L**2     → computes ⟨|C|²⟩/N, wrong exponent
# CORRECT: chi = beta * np.mean(cl) / N   → Wolff improved estimator χ = β⟨|C|⟩/N
#
# Derivation: χ = β·N·⟨m²⟩, and the Wolff cluster estimator for ⟨m²⟩ is
# ⟨m²⟩ = ⟨|C|⟩/N², giving χ = β⟨|C|⟩/N  (Wolff 1989, Eq. 12).
chi = np.array([beta[i] * np.mean(cl) for i, cl in enumerate(cl_list)])

# ── Regression (unchanged — was correct for the given ansatz) ─────────────
# Ansatz: χ(β) = a·(1 − β/β_c)^{−γ}
# Taking logs: log χ = log a − γ·log(1 − β/β_c)
# So: Y = log χ, X = log(1 − β/β_c), slope B = −γ → γ = −B  ✓
X = np.log(1 - beta / beta_c)
Y = np.log(chi)

raw_err = df['sigma_cluster_size'].values
err = raw_err / chi

y = np.sum(Y/(err**2))
x = np.sum(X/(err**2))
sigma = np.sum(1/(err**2))
x2 = np.sum(X**2/(err**2))
xy = np.sum(X*Y/(err**2))

B = (y*x - sigma*xy) / (x**2-sigma*x2)
A = (y - B*x) / sigma


gamma = -B          # correct: slope of log χ vs log(1−β/β_c) is −γ
a = np.exp(A)



# ── Plot ──────────────────────────────────────────────────────────────────
chi_pred = a * (1 - beta / beta_c) ** (-gamma)

delta = sigma*x2 - x**2
variance = sigma / delta
err_B = np.sqrt(variance)  # error in slope B → error in γ is

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Physical space
axes[0].plot(beta/beta_c, chi_pred, 'r-', label=rf'fit $\gamma={gamma:.3f}+/-{err_B:.3f}$')
axes[0].errorbar(beta/beta_c, chi, yerr=raw_err, fmt='o', label='data', zorder=3)
axes[0].set_xlabel(r'$\beta/\beta_c$')
axes[0].set_ylabel(r'$\chi = \beta\langle|C|\rangle/N$')
axes[0].set_title(r'Susceptibility vs. $\beta$')
axes[0].legend()

# Log-log space — should be a straight line if the power law holds
axes[1].plot(np.log(1 - beta/beta_c), np.log(chi_pred), 'r-', label=rf'fit $\gamma={gamma:.3f}+/-{err_B:.3f}$')
axes[1].errorbar(np.log(1 - beta/beta_c), np.log(chi), yerr=err, fmt='o', label='data', zorder=3)
axes[1].set_xlabel(r'$\log(1 - \beta/\beta_c)$')
axes[1].set_ylabel(r'$\log\,\chi$')
axes[1].set_title(r'Log-log: slope $= -\gamma$')
axes[1].legend()

plt.tight_layout()
plt.show()
