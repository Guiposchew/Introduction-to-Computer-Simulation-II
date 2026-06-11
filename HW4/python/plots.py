import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from main import specific_heat, potts_energy
from matplotlib.ticker import NullLocator, ScalarFormatter
   
df= pd.read_csv(f'HW4/python/data/data_q2.csv')
L = df['L'].to_numpy()
tau_sweep = df['tau_sweep'].to_numpy()
tau_cluster = df['tau_cluster'].to_numpy()

coeffs_s = np.polyfit(np.log(L), np.log(tau_sweep), 1)
coeffs_c = np.polyfit(np.log(L), np.log(tau_cluster), 1)

fig, ax = plt.subplots()

plt.title('Autocorreletaion time vs Lattice size')
ax.loglog(L, tau_cluster, 'x', label='simulated data')

ax.loglog(
    L,
    np.exp(coeffs_c[1]) * L**(coeffs_c[0]),
    label=f'fit: y = ({np.exp(coeffs_c[1]):.2f}) * L^{coeffs_c[0]:.2f}'
)

ax.set_xscale('log')

ax.set_xlabel('L')
ax.set_ylabel('Autocorrelation time')

ax.legend()

# ticks at specific values
ticks = [8, 16, 32, 64]
ax.set_xticks(ticks)
ax.set_xticklabels([str(t) for t in ticks])

# remove minor ticks
ax.xaxis.set_minor_locator(NullLocator())

# keep tick formatting clean on log axis
ax.xaxis.set_major_formatter(ScalarFormatter())

plt.show()


fig, ax = plt.subplots()

plt.title('Autocorreletaion time vs Lattice size (rescaled)')
ax.loglog(L, tau_sweep, 'x', label='simulated data')

ax.loglog(
    L,
    np.exp(coeffs_s[1]) * L**(coeffs_s[0]),
    label=f'fit: y = ({np.exp(coeffs_s[1]):.2f}) * L^{coeffs_s[0]:.2f}'
)

ax.set_xscale('log')

ax.set_xlabel('L')
ax.set_ylabel('Autocorrelation time')

ax.legend()

# ticks at specific values
ticks = [8, 16, 32, 64]
ax.set_xticks(ticks)
ax.set_xticklabels([str(t) for t in ticks])

# remove minor ticks
ax.xaxis.set_minor_locator(NullLocator())

# keep tick formatting clean on log axis
ax.xaxis.set_major_formatter(ScalarFormatter())

plt.show()
