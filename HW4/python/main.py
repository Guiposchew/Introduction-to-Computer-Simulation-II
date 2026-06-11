import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

save_dir = "HW4/python/data"

def autocorrelation_time(observable):
    x = np.array(observable)
    x = x - np.mean(x)

    var = np.var(x)
    if var == 0:
        return 0

    acf = np.correlate(x, x, mode='full')[len(x)-1:] / (var * len(x))

    tau_int = 0.5
    for t in range(1, len(acf)):
        if t > 6 * tau_int:
            break
        tau_int += acf[t]

    return tau_int


def get_random_state(L, q):
    return np.random.randint(0, q, size=(L, L), dtype=np.int8)


def wolff_update_potts(state, beta, q, J=1.0):
    L = state.shape[0]

    i0 = np.random.randint(L)
    j0 = np.random.randint(L)
    spin0 = state[i0, j0]

    cluster = np.zeros((L, L), dtype=bool)
    cluster[i0, j0] = True
    stack = [(i0, j0)]

    p_add = 1.0 - np.exp(-beta * J)

    while stack:
        i, j = stack.pop()

        neighbors = [
            ((i + 1) % L, j),
            ((i - 1) % L, j),
            (i, (j + 1) % L),
            (i, (j - 1) % L),
        ]

        for ni, nj in neighbors:
            if not cluster[ni, nj] and state[ni, nj] == spin0:
                if np.random.rand() < p_add:
                    cluster[ni, nj] = True
                    stack.append((ni, nj))

    new_spin = np.random.randint(q)
    while new_spin == spin0:
        new_spin = np.random.randint(q)

    state[cluster] = new_spin
    cluster_size = np.sum(cluster)

    return state, cluster_size


def potts_energy(state, J=1.0):
    L = state.shape[0]
    E = 0

    for i in range(L):
        for j in range(L):
            s = state[i, j]
            E -= J * (s == state[i, (j + 1) % L])
            E -= J * (s == state[(i + 1) % L, j])

    return E


def specific_heat(energies, beta, N):
    E = np.array(energies)
    return (beta**2 / N) * (np.mean(E**2) - np.mean(E)**2)

def magnetization(state):
    counts = np.bincount(state.flatten())
    return np.max(counts)

def run_simulation():

    q = 2
    L_list = [8, 16, 32, 64]
    beta = np.log(1 + np.sqrt(q))

    
    os.makedirs(save_dir, exist_ok=True)

    energies = []
    heats = []

    taus_cluster = []
    taus_sweep = []

    for L in L_list:
        state = get_random_state(L, q)
        N = L * L

        # -----------------------------
        # Equilibration
        # -----------------------------
        n_eq = 5 * N

        for _ in tqdm(range(n_eq), desc=f"Equilibrating L={L}"):
            state, _ = wolff_update_potts(state, beta, q)

        # -----------------------------
        # Measurement
        # -----------------------------
        n_samples = 5000   # increased for stability
        E_list = []
        M_list = []
        cluster_sizes = []

        for _ in tqdm(range(n_samples), desc=f"Measuring L={L}"):

            state, s = wolff_update_potts(state, beta, q)

            E_list.append(potts_energy(state))
            M_list.append(magnetization(state))
            cluster_sizes.append(s)

        # observables
        energies.append(np.mean(E_list))
        heats.append(specific_heat(E_list, beta, N))

        # autocorrelation
        tau_cluster = autocorrelation_time(M_list)
        tau_sweep = tau_cluster * np.mean(cluster_sizes) / N

        taus_cluster.append(tau_cluster)
        taus_sweep.append(tau_sweep)

        print(f"L={L}: tau_cluster={tau_cluster:.2f}, tau_sweep={tau_sweep:.2f}")

    # -----------------------------
    # Save results
    # -----------------------------
    pd.DataFrame({
        "L": L_list,
        "energy": energies,
        "specific_heat": heats,
        "tau_cluster": taus_cluster,
        "tau_sweep": taus_sweep
    }).to_csv(f"{save_dir}/data_q{q}.csv", index=False)

    # -----------------------------
    # Plots
    # -----------------------------
    plt.figure()
    plt.loglog(L_list, taus_sweep, marker='o')
    plt.xlabel("L")
    plt.ylabel("Autocorrelation time (sweeps)")
    plt.title("Wolff autocorrelation time")


    plt.show()


if __name__ == "__main__":
    run_simulation()
