import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

save_dir = "HW5/python/data"
q=2

def equilibrated_state_dir(L):
    return os.path.join(save_dir, f"L{L}")

def equilibrated_state_filename(L, q, beta):
    return os.path.join(equilibrated_state_dir(L), f"Eq/equilibrated_wolff_q{q}_beta{beta:.4f}.npy")

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

def specific_heat(energies, beta, N):
    E = np.array(energies)
    return (beta**2 / N) * (np.mean(E**2) - np.mean(E)**2)

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

    p_add = 1.0 - np.exp(-2*beta * J)

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

def equilibrate_wolff_state(L, q, beta, n_eq=None, force=False):
    filename = equilibrated_state_filename(L, q, beta)
    os.makedirs(equilibrated_state_dir(L), exist_ok=True)
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    if os.path.exists(filename) and not force:
        return np.load(filename)

    state = get_random_state(L, q)
    if n_eq is None:
        n_eq = 5 * L * L

    for _ in tqdm(range(n_eq), desc=f"Equilibrating L={L}"):
        state, _ = wolff_update_potts(state, beta, q)

    np.save(filename, state)
    return state

def load_equilibrated_wolff_state(L, q, beta):
    filename = equilibrated_state_filename(L, q, beta)
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Equilibrated state not found: {filename}")
    return np.load(filename)

def measure_wolff_observables(state, beta, q, n_samples=5000):
    N = state.size
    energies = []
    cluster_sizes = []

    for _ in tqdm(range(n_samples), desc="Measuring"):
        state, s = wolff_update_potts(state, beta, q)
        energies.append(potts_energy(state))
        cluster_sizes.append(s)

    energies = np.array(energies)
    cluster_sizes = np.array(cluster_sizes)

    # Compute standard deviation of raw samples
    sigma_S = np.std(cluster_sizes, ddof=1)
    
    # 🔥 FIX: Compute the integrated autocorrelation time for cluster sizes
    tau_S = autocorrelation_time(cluster_sizes)
    
    # 🔥 FIX: Compute the true standard error of the mean accounting for correlation
    # If tau_S is small (e.g. near 0.5), 2*tau_S ≈ 1, reverting to independent samples.
    sigma_mean_S = sigma_S * np.sqrt((2 * tau_S) / n_samples)

    return {
        "energy": np.mean(energies),
        "specific_heat": specific_heat(energies, beta, N),
        "mean_cluster_size": np.mean(cluster_sizes),
        "cluster_sizes": cluster_sizes,
        "sigma_cluster_size": sigma_mean_S,  # Now contains the true error bar
    }
    

def measurement_filename(L, q, beta):
    return os.path.join(equilibrated_state_dir(L), f"measurement_wolff_q{q}_beta{beta:.4f}.csv")


def measure_wolff_observables_from_file(L, q, beta, n_samples=1000, save=True):
    state = load_equilibrated_wolff_state(L, q, beta)
    results = measure_wolff_observables(state, beta, q, n_samples=n_samples)

    if save:
        os.makedirs(equilibrated_state_dir(L), exist_ok=True)
        df = pd.DataFrame([{ 
            "L": L,
            "q": q,
            "Beta": beta,
            "energy": results["energy"],
            "specific_heat": results["specific_heat"],
            "mean_cluster_size": results["mean_cluster_size"],
            "cluster_sizes": results["cluster_sizes"],
            "sigma_cluster_size": results["sigma_cluster_size"],
        }])
        df.to_csv(measurement_filename(L, q, beta), index=False)

    return results

def run_simulation(beta, q=q):
    L_list = [8, 16, 32, 64]
    os.makedirs(save_dir, exist_ok=True)

    energies = []
    heats = []
    mean_cluster_size = []

    for L in L_list:
        state = equilibrate_wolff_state(L, q, beta)
        results = measure_wolff_observables(state, beta, q)

        energies.append(results["energy"])
        heats.append(results["specific_heat"])
        mean_cluster_size.append(results["mean_cluster_size"])

    pd.DataFrame({
        "L": L_list,
        "Beta": beta,
        "energy": energies,
        "specific_heat": heats,
        "mean_cluster_size": mean_cluster_size,
    }).to_csv(f"{save_dir}/data_q{(beta/np.log(1 + np.sqrt(2))):.2f}.csv", index=False)


if __name__ == "__main__":
    for i in range(10, 11, 1):
        beta = np.log(1 + np.sqrt(q)) * i / 10
        for L in [8, 16, 32, 64]:
            measure_wolff_observables_from_file(L, q, beta)
