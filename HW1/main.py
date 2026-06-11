import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


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


def get_magnetization(state):
    """
    Compute the magnetization per spin.
    """
    return float(np.mean(state))


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
        state, _ = wolff_update(state, beta, J)

    energy_history = np.zeros(n_measurements, dtype=float)
    magnetization_history = np.zeros(n_measurements, dtype=float)

    for m in range(n_measurements):
        flipped_spins = 0

        # Do enough cluster flips so that on average
        # about one lattice worth of spins is updated.
        while flipped_spins < N:
            state, cluster_size = wolff_update(state, beta, J)
            flipped_spins += cluster_size

        energy_history[m] = get_energy(state, J=J, h=h)
        magnetization_history[m] = get_magnetization(state)

    return state, energy_history, magnetization_history


def simulate_lattice(
    L,
    betas,
    J=1.0,
    h=0.0,
    n_repeats=5,
    n_thermalization=100,
    n_measurements=500,
):
    """
    Simulate one lattice size across a range of beta values.

    Returns a dictionary with:
    - betas
    - mean energy per spin
    - mean |magnetization|
    - Binder cumulant
    """
    E_mean = np.zeros(len(betas))
    M_abs_mean = np.zeros(len(betas))
    M2_mean = np.zeros(len(betas))
    M4_mean = np.zeros(len(betas))
    binder = np.zeros(len(betas))

    for b_idx, beta in enumerate(tqdm(betas, desc=f"L={L}", leave=False)):
        E_samples_all = []
        M_samples_all = []

        # Start from a fresh random state for each repeat
        for _ in range(n_repeats):
            state = get_random_state(L)

            _, E_hist, M_hist = wolff_run(
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
        M2_mean[b_idx] = np.mean(M_samples_all**2)
        M4_mean[b_idx] = np.mean(M_samples_all**4)

        if M2_mean[b_idx] > 0:
            binder[b_idx] = 1.0 - M4_mean[b_idx] / (3.0 * M2_mean[b_idx]**2)
        else:
            binder[b_idx] = np.nan

    return {
        "L": L,
        "betas": betas,
        "energy_per_spin": E_mean,
        "magnetization": M_samples_all,
        "binder": binder,
    }


def main():
    # -----------------------------------------------------
    # PARAMETERS
    # -----------------------------------------------------
    J = 1.0
    h = 0.0                  # external field remains variable
    lattice_sizes = [8, 16, 32, 64]
    beta_c = np.log(1 + np.sqrt(2)) / 2

    # More beta values, concentrated around the critical region
    betas = np.concatenate([
        np.linspace(0.1, 0.35, 12, endpoint=False),
        np.linspace(0.35, 0.55, 30, endpoint=False),
        np.linspace(0.55, 0.8, 12)
    ])

    n_repeats = 4
    n_thermalization = 100
    n_measurements = 400

    # -----------------------------------------------------
    # RUN SIMULATIONS
    # -----------------------------------------------------
    results = []

    for L in tqdm(lattice_sizes, desc="Lattice sizes"):
        res = simulate_lattice(
            L=L,
            betas=betas,
            J=J,
            h=h,
            n_repeats=n_repeats,
            n_thermalization=n_thermalization,
            n_measurements=n_measurements,
        )
        results.append(res)

    # -----------------------------------------------------
    # PLOTS
    # -----------------------------------------------------
    plt.figure(figsize=(8, 5))
    for res in results:
        plt.plot(
            res["betas"] / beta_c,
            res["binder"],
            marker="o",
            ms=3,
            label=f"L={res['L']}"
        )
    plt.xlabel(r"$\beta / \beta_c$")
    plt.ylabel("Binder cumulant")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    for res in results:
        plt.plot(
            res["betas"] / beta_c,
            res["magnetization_abs"],
            marker="o",
            ms=3,
            label=f"L={res['L']}"
        )
    plt.xlabel(r"$\beta / \beta_c$")
    plt.ylabel(r"$\langle |m| \rangle$")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    for res in results:
        plt.plot(
            res["betas"] / beta_c,
            res["energy_per_spin"],
            marker="o",
            ms=3,
            label=f"L={res['L']}"
        )
    plt.xlabel(r"$\beta / \beta_c$")
    plt.ylabel(r"$\langle E \rangle / N$")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()