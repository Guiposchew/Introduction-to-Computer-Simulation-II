import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from scipy import ndimage as nd
import scipy



def get_Random_state(X):
    return np.random.choice([-1, 1], size=X)

def metropolis_sweep(state_0, beta):
    state = state_0.copy()
    Lx, Ly = state.shape
    for _ in range(Lx * Ly):
        i = np.random.randint(0, Lx)
        j = np.random.randint(0, Ly)

        s = state[i, j]
        nb = (
            state[(i+1)%Lx, j] +
            state[(i-1)%Lx, j] +
            state[i, (j+1)%Ly] +
            state[i, (j-1)%Ly]
        )

        dE = 2 * s * nb

        if dE <= 0 or np.random.rand() < np.exp(-beta * dE):
            state[i, j] = -s

    return state


L = np.array([64,64])

state_rand = get_Random_state(L)
state_ordered = np.ones(L)

states = [state_rand, state_ordered]

betas =  [3.0, 2.5, 2.269185, 2.0, 1.5]

state_rand_new = []
state_ordered_new = []

sweeps = 1e4


for j in range(len(betas)):
    if betas[j] >= 2.269185:
        current_state = state_rand.copy()
        for _ in tqdm(range(int(sweeps)), desc=f"Random beta={betas[j]}"):
            current_state = metropolis_sweep(current_state, beta=betas[j])
        state_rand_new.append(current_state)
    else:
        current_state = state_ordered.copy()
        for _ in tqdm(range(int(sweeps)), desc=f"Ordered beta={betas[j]}"):
            current_state = metropolis_sweep(current_state, beta=betas[j])
        state_ordered_new.append(current_state)

rand_idx = 0
ordered_idx = 0
for i in range(len(betas)):
    if betas[i] >= 2.269185:
        plt.figure(f'Random {i}')
        plt.imshow(state_rand_new[rand_idx], cmap='RdBu', interpolation='nearest', vmin=-1, vmax=1)
        plt.colorbar(label='Spin')
        plt.title(f"Final Lattice Configuration (Random, beta={betas[i]})")
        rand_idx += 1
    else:
        plt.figure(f'Ordered {i}')
        plt.imshow(state_ordered_new[ordered_idx], cmap='RdBu', interpolation='nearest', vmin=-1, vmax=1)
        plt.colorbar(label='Spin')
        plt.title(f"Final Lattice Configuration (Ordered, beta={betas[i]})")
        ordered_idx += 1

plt.show()