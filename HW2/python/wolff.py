import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from numba import njit
import os


# =========================================================
# 2D ISING MODEL: WOLFF CLUSTER MONTE CARLO
# =========================================================

def get_random_state(L):
    """
    Create a random L x L Ising spin configuration with values ±1.
    """
    return np.random.choice([-1, 1], size=(L, L)).astype(np.int8)


def get_energy(state, J=1.0, h=0.0):
    """
    Compute the total energy of a 2D Ising configuration
    with periodic boundary conditions.

    E = -J * sum_<ij> s_i s_j - h * sum_i s_i
    """
    right = np.roll(state, -1, axis=1)
    down = np.roll(state, -1, axis=0)
    return float(-J * np.sum(state * (right + down)) - h * np.sum(state))


@njit
def get_energy_fast(state, J=1.0, h=0.0):
    """
    Fast energy computation using loops for Numba compatibility.
    """
    L = state.shape[0]
    energy = 0.0
    for i in range(L):
        for j in range(L):
            energy -= J * state[i, j] * (state[(i+1) % L, j] + state[i, (j+1) % L])
    energy -= h * np.sum(state)
    return energy


def get_magnetization(state):
    """
    Compute the magnetization per spin.
    """
    return float(np.mean(state))


def compute_autocorr_time(series):
    """
    Compute the integrated autocorrelation time using the binning method.
    """
    series = np.array(series, dtype=float)
    n = len(series)
    if n < 2:
        return 0.0
    var1 = np.var(series, ddof=1)
    if var1 == 0:
        return 0.0
    max_bins = int(np.log2(n)) - 1
    bins = [series]
    variances = [var1]
    for k in range(1, max_bins + 1):
        current = bins[-1]
        if len(current) < 2:
            break
        new_bin = []
        for i in range(0, len(current) - 1, 2):
            new_bin.append((current[i] + current[i+1]) / 2)
        new_bin = np.array(new_bin)
        bins.append(new_bin)
        variances.append(np.var(new_bin, ddof=1))
    var_inf = variances[-1]
    tau = (var1 / var_inf - 1) / 2
    return tau


def metropolis_update(state, beta, J=1.0):
    """
    Perform one Metropolis sweep (one update per spin on average).
    """
    L = state.shape[0]
    for _ in range(L * L):
        i = np.random.randint(0, L)
        j = np.random.randint(0, L)
        s = state[i, j]
        nb = (
            state[(i+1) % L, j] +
            state[(i-1) % L, j] +
            state[i, (j+1) % L] +
            state[i, (j-1) % L]
        )
        dE = 2 * J * s * nb
        if dE <= 0 or np.random.random() < np.exp(-beta * dE):
            state[i, j] = -s
    return state


def metropolis_run(
    state,
    beta,
    J=1.0,
    h=0.0,
    n_thermalization=100,
    n_measurements=500,
):
    """
    Run a Metropolis Monte Carlo simulation at fixed beta.

    Parameters
    ----------
    state : np.ndarray
        Initial configuration.
    beta : float
        Inverse temperature.
    J : float
        Coupling constant.
    h : float
        External field.
    n_thermalization : int
        Number of sweeps used for equilibration.
    n_measurements : int
        Number of measurements to record.

    Returns
    -------
    final_state : np.ndarray
    energy_history : np.ndarray
    magnetization_history : np.ndarray
    """
    state = state.copy()
    L = state.shape[0]
    N = L * L

    # Thermalization
    for _ in range(n_thermalization):
        state = metropolis_update(state, beta, J)

    energy_history = np.zeros(n_measurements, dtype=float)
    magnetization_history = np.zeros(n_measurements, dtype=float)

    for m in range(n_measurements):
        state = metropolis_update(state, beta, J)
        energy_history[m] = get_energy_fast(state, J=J, h=h)
        magnetization_history[m] = get_magnetization(state)

    return state, energy_history, magnetization_history


def wolff_update(state, beta, J=1.0):
    """
    Perform one Wolff cluster update.

    Returns
    -------
    state : np.ndarray
        Updated spin configuration.
    cluster_size : int
        Number of spins flipped in this cluster update.
    """
    L = state.shape[0]
    p_add = 1.0 - np.exp(-2.0 * beta * J)

    i0 = np.random.randint(0, L)
    j0 = np.random.randint(0, L)

    cluster_spin = state[i0, j0]

    stack = [(i0, j0)]
    in_cluster = np.zeros((L, L), dtype=bool)
    in_cluster[i0, j0] = True

    cluster_sites = []

    while stack:
        i, j = stack.pop()
        cluster_sites.append((i, j))

        neighbors = [
            ((i + 1) % L, j),
            ((i - 1) % L, j),
            (i, (j + 1) % L),
            (i, (j - 1) % L),
        ]

        for ni, nj in neighbors:
            if not in_cluster[ni, nj] and state[ni, nj] == cluster_spin:
                if np.random.random() < p_add:
                    in_cluster[ni, nj] = True
                    stack.append((ni, nj))

    for i, j in cluster_sites:
        state[i, j] *= -1

    return state, len(cluster_sites)


@njit
def wolff_update_fast(state, beta, J=1.0):
    """
    Optimized Wolff cluster update using flat indexing and Numba.
    """
    L = state.shape[0]
    N = L * L
    p_add = 1.0 - np.exp(-2.0 * beta * J)

    i0 = np.random.randint(0, L)
    j0 = np.random.randint(0, L)
    idx0 = i0 * L + j0

    cluster_spin = state.flat[idx0]

    stack = [idx0]
    in_cluster = np.zeros(N, dtype=np.bool_)
    in_cluster[idx0] = True
    cluster_size = 1

    while len(stack) > 0:
        idx = stack.pop()
        i = idx // L
        j = idx % L

        # Neighbors
        neighbors = [
            ((i + 1) % L) * L + j,
            ((i - 1) % L) * L + j,
            i * L + (j + 1) % L,
            i * L + (j - 1) % L,
        ]

        for nidx in neighbors:
            if not in_cluster[nidx] and state.flat[nidx] == cluster_spin:
                if np.random.random() < p_add:
                    in_cluster[nidx] = True
                    stack.append(nidx)
                    cluster_size += 1

    # Flip all in cluster
    for idx in range(N):
        if in_cluster[idx]:
            state.flat[idx] = -state.flat[idx]

    return state, cluster_size


def wolff_run(
    state,
    beta,
    J=1.0,
    h=0.0,
    n_thermalization=100,
    n_measurements=500,
):
    """
    Run a Wolff Monte Carlo simulation at fixed beta.

    We thermalize first, then collect measurements.
    Since one Wolff update flips a variable number of spins,
    we space measurements so that roughly one "sweep" worth of spins
    is updated between measurements.

    Parameters
    ----------
    state : np.ndarray
        Initial configuration.
    beta : float
        Inverse temperature.
    J : float
        Coupling constant.
    h : float
        External field.
    n_thermalization : int
        Number of Wolff updates used for equilibration.
    n_measurements : int
        Number of measurements to record.

    Returns
    -------
    final_state : np.ndarray
    energy_history : np.ndarray
    magnetization_history : np.ndarray
    """
    state = state.copy()
    L = state.shape[0]
    N = L * L

    # Thermalization
    for _ in range(n_thermalization):
        state, _ = wolff_update_fast(state, beta, J)

    energy_history = np.zeros(n_measurements, dtype=float)
    magnetization_history = np.zeros(n_measurements, dtype=float)

    for m in range(n_measurements):
        flipped_spins = 0

        # Do enough cluster flips so that on average
        # about one lattice worth of spins is updated.
        while flipped_spins < N:
            state, cluster_size = wolff_update_fast(state, beta, J)
            flipped_spins += cluster_size

        energy_history[m] = get_energy_fast(state, J=J, h=h)
        magnetization_history[m] = get_magnetization(state)

    return state, energy_history, magnetization_history


def simulate_lattice(
    L,
    betas,
    J=1.0,
    h=0.0,
    method='wolff',
    n_repeats=5,
    n_thermalization=100,
    n_measurements=500,
):
    """
    Simulate one lattice size across a range of beta values using the specified method.

    Returns a dictionary with:
    - betas
    - mean energy per spin
    - mean |magnetization|
    - specific heat per spin
    - Binder cumulant
    - autocorrelation times for energy and magnetization
    """
    E_mean = np.zeros(len(betas))
    M_abs_mean = np.zeros(len(betas))
    C_mean = np.zeros(len(betas))
    M2_mean = np.zeros(len(betas))
    M4_mean = np.zeros(len(betas))
    binder = np.zeros(len(betas))
    tau_energy = np.zeros(len(betas))
    tau_magnetization = np.zeros(len(betas))

    for b_idx, beta in enumerate(tqdm(betas, desc=f"L={L} {method}", leave=False)):
        E_samples_all = []
        M_samples_all = []

        # Start from a fresh random state for each repeat
        for _ in range(n_repeats):
            state = get_random_state(L)

            if method == 'wolff':
                _, E_hist, M_hist = wolff_run(
                    state=state,
                    beta=beta,
                    J=J,
                    h=h,
                    n_thermalization=n_thermalization,
                    n_measurements=n_measurements,
                )
            else:
                _, E_hist, M_hist = metropolis_run(
                    state=state,
                    beta=beta,
                    J=J,
                    h=h,
                    n_thermalization=n_thermalization,
                    n_measurements=n_measurements,
                )

            E_samples_all.extend(E_hist)
            M_samples_all.extend(M_hist)

        E_samples_all = np.array(E_samples_all, dtype=float)
        M_samples_all = np.array(M_samples_all, dtype=float)

        N = L * L

        E_mean[b_idx] = np.mean(E_samples_all) / N
        M_abs_mean[b_idx] = np.mean(np.abs(M_samples_all))
        E2_mean = np.mean(E_samples_all**2)
        E_mean_val = np.mean(E_samples_all)
        C_mean[b_idx] = beta**2 * (E2_mean - E_mean_val**2) / N
        M2_mean[b_idx] = np.mean(M_samples_all**2)
        M4_mean[b_idx] = np.mean(M_samples_all**4)
        if M2_mean[b_idx] > 0:
            binder[b_idx] = 1.0 - M4_mean[b_idx] / (3.0 * M2_mean[b_idx]**2)
        else:
            binder[b_idx] = np.nan

        tau_energy[b_idx] = compute_autocorr_time(E_samples_all)
        tau_magnetization[b_idx] = compute_autocorr_time(np.abs(M_samples_all))

    return {
        "L": L,
        "method": method,
        "betas": betas,
        "energy_per_spin": E_mean,
        "magnetization_abs": M_abs_mean,
        "specific_heat": C_mean,
        "binder": binder,
        "tau_energy": tau_energy,
        "tau_magnetization": tau_magnetization,
    }


def main():
    # -----------------------------------------------------
    # PARAMETERS
    # -----------------------------------------------------
    J = 1.0
    h = 0.0                  # external field remains variable
    lattice_sizes = [8, 16, 32, 64]
    beta_c = np.log(1 + np.sqrt(2)) / 2
    T_c = 1 / beta_c

    # 10 betas around critical temperature
    betas = np.linspace(0.35, 0.55, 10)

    n_repeats = 4
    n_thermalization = 100
    n_measurements = 400

    methods = ['wolff']

    # -----------------------------------------------------
    # RUN SIMULATIONS
    # -----------------------------------------------------
    results = {method: [] for method in methods}

    for method in methods:
        for L in tqdm(lattice_sizes, desc=f"{method} Lattice sizes"):
            res = simulate_lattice(
                L=L,
                betas=betas,
                J=J,
                h=h,
                method=method,
                n_repeats=n_repeats,
                n_thermalization=n_thermalization,
                n_measurements=n_measurements,
            )
            results[method].append(res)

    # -----------------------------------------------------
    # SAVE RESULTS TO CSV
    # -----------------------------------------------------
    os.makedirs('data', exist_ok=True)
    for method in methods:
        for res in results[method]:
            L = res['L']
            filename = f"data/{method}_L{L}.csv"
            df = pd.DataFrame({
                'beta': res['betas'],
                'T': 1 / res['betas'],
                'T_over_T_c': (1 / res['betas']) / T_c,
                'energy_per_spin': res['energy_per_spin'],
                'magnetization_abs': res['magnetization_abs'],
                'specific_heat': res['specific_heat'],
                'binder': res['binder'],
                'tau_energy': res['tau_energy'],
                'tau_magnetization': res['tau_magnetization'],
            })
            df.to_csv(filename, index=False)
            print(f"Saved {filename}")

    # -----------------------------------------------------
    # AUTOCORRELATION TIMES AT CRITICAL POINT
    # -----------------------------------------------------
    idx_c = np.argmin(np.abs(betas - beta_c))
    print(f"Autocorrelation times at beta ≈ {betas[idx_c]:.3f} (beta_c ≈ {beta_c:.3f})")
    for method in methods:
        for res in results[method]:
            tau_E = res['tau_energy'][idx_c]
            tau_M = res['tau_magnetization'][idx_c]
            print(f"{res['method']} L={res['L']}: tau_E={tau_E:.2f}, tau_M={tau_M:.2f}")

    # -----------------------------------------------------
    # PLOTS
    # -----------------------------------------------------
    for method in methods:
        plt.figure(figsize=(8, 5))
        for res in results[method]:
            T_over_Tc = (1 / res["betas"]) / T_c
            plt.plot(
                T_over_Tc,
                res["binder"],
                marker="o",
                ms=3,
                label=f"L={res['L']}"
            )
        plt.xlabel(r"$T / T_c$")
        plt.ylabel("Binder cumulant")
        plt.title(f"Binder cumulant - {method}")
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(8, 5))
        for res in results[method]:
            T_over_Tc = (1 / res["betas"]) / T_c
            plt.plot(
                T_over_Tc,
                res["magnetization_abs"],
                marker="o",
                ms=3,
                label=f"L={res['L']}"
            )
        plt.xlabel(r"$T / T_c$")
        plt.ylabel(r"$\langle |m| \rangle$")
        plt.title(f"Magnetization - {method}")
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(8, 5))
        for res in results[method]:
            T_over_Tc = (1 / res["betas"]) / T_c
            plt.plot(
                T_over_Tc,
                res["energy_per_spin"],
                marker="o",
                ms=3,
                label=f"L={res['L']}"
            )
        plt.xlabel(r"$T / T_c$")
        plt.ylabel(r"$\langle E \rangle / N$")
        plt.title(f"Energy per spin - {method}")
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(8, 5))
        for res in results[method]:
            T_over_Tc = (1 / res["betas"]) / T_c
            plt.plot(
                T_over_Tc,
                res["specific_heat"],
                marker="o",
                ms=3,
                label=f"L={res['L']}"
            )
        plt.xlabel(r"$T / T_c$")
        plt.ylabel(r"$C / N$")
        plt.title(f"Specific heat per spin - {method}")
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()