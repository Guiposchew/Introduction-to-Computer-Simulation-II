import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os


# -----------------------------
# Utilities
# -----------------------------
def get_random_state(L, q):
    return np.random.randint(0, q, size=(L, L), dtype=np.int8)


# -----------------------------
# Wolff update (correct Potts)
# -----------------------------
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
            if not cluster[ni, nj]:
                if state[ni, nj] == spin0:
                    if np.random.rand() < p_add:
                        cluster[ni, nj] = True
                        stack.append((ni, nj))

    new_spin = np.random.randint(q)
    while new_spin == spin0:
        new_spin = np.random.randint(q)

    state[cluster] = new_spin
    return state


# -----------------------------
# Observables
# -----------------------------
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


# -----------------------------
# Main simulation
# -----------------------------
def run_simulation():

    q = 3
    L_list = [4, 8, 16, 32, 64, 128]
    beta = np.log(1 + np.sqrt(q))

    save_dir = "HW3/python/data"
    os.makedirs(save_dir, exist_ok=True)

    energies = []
    heats = []

    states_final = []

    for L in L_list:

        state = get_random_state(L, q)
        N = L * L

        # -----------------------------
        # Equilibration
        # -----------------------------
        n_eq = 5 * N  # safe for Wolff scaling

        for _ in tqdm(range(n_eq), desc=f"Equilibrating L={L}"):
            state = wolff_update_potts(state, beta, q)

        # -----------------------------
        # Measurement
        # -----------------------------
        n_samples = 1000
        E_list = []

        for _ in tqdm(range(n_samples), desc=f"Measuring L={L}"):
            state = wolff_update_potts(state, beta, q)
            E_list.append(potts_energy(state))

        # store observables
        energies.append(np.mean(E_list))
        heats.append(specific_heat(E_list, beta, N))

        states_final.append(state.copy())

    # -----------------------------
    # Save results
    # # -----------------------------
    # np.save(f"{save_dir}/final_states_q{q}.csv", np.array(states_final, dtype=object))
    # np.save(f"{save_dir}/energies_q{q}.csv", np.array(energies))
    # np.save(f"{save_dir}/specific_heat_q{q}.csv", np.array(heats))


    # optional CSV
    pd.DataFrame({
        "L": L_list,
        "energy": energies,
        "specific_heat": heats,
        "states": states_final
    }).to_csv(f"{save_dir}/data{q}.csv", index=False)

    # -----------------------------
    # Plots
    # -----------------------------
    plt.figure("Energy")
    plt.plot(L_list, energies, marker='o')
    plt.xlabel("L")
    plt.ylabel("Energy")
    plt.title("Energy vs L at βc")

    plt.figure("Specific Heat")
    plt.plot(L_list, heats, marker='o')
    plt.xlabel("L")
    plt.ylabel("C")
    plt.title("Specific Heat vs L at βc")

    plt.show()


if __name__ == "__main__":
    run_simulation()
