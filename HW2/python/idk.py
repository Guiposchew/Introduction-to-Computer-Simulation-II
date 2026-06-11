import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

files = [
    "HW2/data/wolff_L8.csv",
    "HW2/data/wolff_L16.csv",
    "HW2/data/wolff_L32.csv",
    "HW2/data/wolff_L64.csv"
]

dfs = []

for f in files:
    df = pd.read_csv(f)

    # extract L from filename
    L = int(f.split("L")[1].split(".")[0])

    df["L"] = L
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)

df_tc = df[np.isclose(df["T_over_T_c"], 1.0, atol=0.05)]

tau_E = df_tc.groupby("L")["tau_energy"].mean()
tau_M = df_tc.groupby("L")["tau_magnetization"].mean()

def fit_z(series):
    L = series.index.values
    tau = series.values

    logL = np.log(L)
    logT = np.log(tau)

    slope, intercept, r, p, err = stats.linregress(logL, logT)
    return slope, err, r**2

z_E, err_E, r2_E = fit_z(tau_E)
z_M, err_M, r2_M = fit_z(tau_M)

print("z (energy):", z_E, "+/-", err_E, "R²:", r2_E)
print("z (magnetization):", z_M, "+/-", err_M, "R²:", r2_M)

plt.figure()

plt.loglog(tau_E.index, tau_E.values, 'o-', label="Energy")
plt.loglog(tau_M.index, tau_M.values, 'o-', label="Magnetization")

plt.xlabel("L")
plt.ylabel("tau")
plt.legend()
plt.show()