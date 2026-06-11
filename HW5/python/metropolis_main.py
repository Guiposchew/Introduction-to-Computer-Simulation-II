import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

save_dir = "HW5/python/data"

def equilibrated_state_dir(L):
    return os.path.join(save_dir, f"L{L}")

def equilibrated_state_filename(L, q, beta):
    return os.path.join(equilibrated_state_dir(L), f"Eq/equilibrated_wolff_q{q}_beta{beta:.4f}.npy")

def measurement_filename(L, q, beta):
    return os.path.join(equilibrated_state_dir(L), f"measurement_metropolis_q{q}_beta{beta:.4f}.csv")

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

def metropolis_update_potts(state, beta, q, J=1.0):
    L = state.shape[0]
    for _ in range(L * L):  # one sweep
        i = np.random.randint(L)
        j = np.random.randint(L)
        s_old = state[i, j]

        # choose new spin != s_old
        possible = [k for k in range(q) if k != s_old]
        s_new = np.random.choice(possible)

        # compute delta E
        neighbors = [
            ((i + 1) % L, j),
            ((i - 1) % L, j),
            (i, (j + 1) % L),
            (i, (j - 1) % L),
        ]
        num_same_old = sum(1 for ni, nj in neighbors if state[ni, nj] == s_old)
        num_same_new = sum(1 for ni, nj in neighbors if state[ni, nj] == s_new)
        delta_E = J * (num_same_old - num_same_new)

        if delta_E <= 0 or np.random.rand() < np.exp(-beta * delta_E):
            state[i, j] = s_new

    return state

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
    counts = np.bincount(state.flatten(), minlength=2)  # assuming q=2 for bincount
    return np.max(counts)

def susceptibility(M_list, beta, N):
    M = np.array(M_list)
    return (beta / N) * (np.mean(M**2) - np.mean(M)**2)

def equilibrate_metropolis_state(L, q, beta, n_eq=None, force=False):
    # Load precalculated Wolff equilibrated state
    filename = equilibrated_state_filename(L, q, beta)
    if os.path.exists(filename):
        return np.load(filename)
    else:
        raise FileNotFoundError(f"Precalculated Wolff equilibrated state not found for L={L}, q={q}, beta={beta:.4f}. Run Wolff equilibration first.")

def load_equilibrated_metropolis_state(L, q, beta):
    filename = equilibrated_state_filename(L, q, beta)
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Equilibrated state not found: {filename}")
    return np.load(filename)

def measure_metropolis_observables(state, beta, q, n_samples=5000):
    N = state.size
    energies = []
    magnetizations = []

    for _ in tqdm(range(n_samples), desc="Measuring"):
        state = metropolis_update_potts(state, beta, q)
        energies.append(potts_energy(state))
        magnetizations.append(magnetization(state))

    return {
        "energy": np.mean(energies),
        "specific_heat": specific_heat(energies, beta, N),
        "mean_magnetization": np.mean(magnetizations),
        "susceptibility": susceptibility(magnetizations, beta, N),
    }

def measure_metropolis_observables_from_file(L, q, beta, n_samples=1000, save=True):
    state = load_equilibrated_metropolis_state(L, q, beta)
    results = measure_metropolis_observables(state, beta, q, n_samples=n_samples)

    if save:
        os.makedirs(equilibrated_state_dir(L), exist_ok=True)
        df = pd.DataFrame([{
            "L": L,
            "q": q,
            "Beta": beta,
            "energy": results["energy"],
            "specific_heat": results["specific_heat"],
            "mean_magnetization": results["mean_magnetization"],
            "susceptibility": results["susceptibility"],
        }])
        df.to_csv(measurement_filename(L, q, beta), index=False)

    return results

def run_simulation(q=2, beta=np.log(1 + np.sqrt(2))):
    L_list = [8, 16, 32, 64]
    os.makedirs(save_dir, exist_ok=True)

    energies = []
    heats = []
    magnetizations = []
    susceptibilities = []

    for L in L_list:
        state = get_random_state(L, q)
        N = L * L

        # Equilibration
        n_eq = 5 * N
        for _ in tqdm(range(n_eq), desc=f"Equilibrating L={L}"):
            state = metropolis_update_potts(state, beta, q)

        # Measurement
        n_samples = 5000
        E_list = []
        M_list = []

        for _ in tqdm(range(n_samples), desc=f"Measuring L={L}"):
            state = metropolis_update_potts(state, beta, q)
            E_list.append(potts_energy(state))
            M_list.append(magnetization(state))

        # observables
        energies.append(np.mean(E_list))
        heats.append(specific_heat(E_list, beta, N))
        magnetizations.append(np.mean(M_list))
        susceptibilities.append(susceptibility(M_list, beta, N))

def run_simulation(q=2, beta=np.log(1 + np.sqrt(2))):
    L_list = [8, 16, 32, 64]
    os.makedirs(save_dir, exist_ok=True)

    energies = []
    heats = []
    magnetizations = []
    susceptibilities = []

    for L in L_list:
        state = equilibrate_metropolis_state(L, q, beta)
        results = measure_metropolis_observables(state, beta, q)

        energies.append(results["energy"])
        heats.append(results["specific_heat"])
        magnetizations.append(results["mean_magnetization"])
        susceptibilities.append(results["susceptibility"])

    pd.DataFrame({
        'L': L_list,
        'energy': energies,
        'specific_heat': heats,
        'mean_magnetization': magnetizations,
        'susceptibility': susceptibilities,
        'Beta': [beta] * len(L_list)
    }).to_csv(os.path.join(save_dir, f'data_metropolis_q{beta:.2f}.csv'), index=False)

if __name__ == "__main__":
    for i in range(1, 11):
        beta = np.log(1 + np.sqrt(2)) * i / 10
        for L in [8, 16, 32, 64]:
            measure_metropolis_observables_from_file(L, 2, beta)
