import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from main import specific_heat, potts_energy

dudbeta = []

for i in range(4):    
    df= pd.read_csv(f'HW1/data/ising_L{2**(i+3)}.csv')
    U = df['binder'].to_numpy()
    beta = df['beta'].to_numpy()
    dudbeta.append(np.gradient(U,beta)[26])

L = [8,16,32,64]

plt.figure('dudbeta_vs_L')
plt.title(r'$dU/d\beta$ vs $L$ at $\beta_c$')
plt.xlabel(r'$L$')
plt.ylabel(r'$dU/d\beta$')
plt.plot(L,dudbeta, marker='o' ,label = 'Simulated results')
plt.legend()
plt.show()

# C = df['specific_heat'].to_numpy()
# L = df['L'].to_numpy()

# plt.figure('specific_heat')
# plt.plot(np.log(L), np.log(C), marker='x')
# plt.plot(np.log(L), coeffs[0]*np.log(L) + coeffs[1], label=f'Fit: ln(C) = ln(L){coeffs[0]:.2f}')
# plt.xlabel('Lattice Size L')
# plt.ylabel('Specific Heat C')
# plt.title('Specific Heat vs Lattice Size')
# plt.legend()

# E = df['energy'].to_numpy()


# coeffs = np.polyfit(np.log(L), np.log(np.abs(E)), 1)

# plt.figure('energy')
# plt.plot(np.log(L),np.log(np.abs(E)), marker='x')
# plt.plot(np.log(L), np.log(L)*coeffs[0]+coeffs[1], label = f'Fit: ln(E)=ln(L)x{coeffs[0]:.2f} + {coeffs[1]:.2f}')
# plt.xlabel('Lattice Size L')
# plt.ylabel('Energy E')
# plt.title('Energy vs Lattice Size')
# plt.legend()

# plt.show()



# df = pd.read_csv('HW3/python/data/potts_states.csv')

# L = np.fromstring(df['L'][0].strip('[]'), sep=',').astype(int)
# states = []

# beta = df['beta'][0]
# q= df['q'][0]

# for i in range(len(df['state'])):
#     state = np.fromstring(df['state'][i], sep=' ').reshape(L[i], L[i])
#     states.append(state)

# energies = [potts_energy(state) for state in states]

# specific_heats = [specific_heat(energies, beta, L[i]**2) for i, state in enumerate(states)]

# plt.figure('specific_heat')
# plt.plot(L, specific_heats, marker='o')
# plt.xlabel('Lattice Size L')
# plt.ylabel('Specific Heat C')
# plt.title('Specific Heat vs Lattice Size')
# #plt.xscale('log')

# plt.figure('energy')
# plt.plot(L, energies, marker='o')
# plt.xlabel('Lattice Size L')
# plt.ylabel('Energy E')
# plt.title('Energy vs Lattice Size')

# fig, axes = plt.subplots(2, 3, figsize=(10, 6), constrained_layout=True)
# axes = axes.flatten()

# vmin, vmax = 0, q-1

# for i in range(len(L)):
#     im = axes[i].imshow(states[i], cmap='viridis',
#                         interpolation='nearest',
#                         vmin=vmin, vmax=vmax)
#     axes[i].set_title(f'L={L[i]}')
#     axes[i].axis('off')

# # shared colorbar with padding
# cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
# cbar.set_label("Spin")

# plt.show()