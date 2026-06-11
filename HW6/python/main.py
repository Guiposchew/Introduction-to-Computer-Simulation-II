import os
from pathlib import Path

import numpy as np
import pandas as pd

SAVE_DIR = Path(__file__).resolve().parent / "data"


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def random_state(L):
    return np.random.uniform(0.0, 2.0 * np.pi, size=(L, L))


def neighbor_indices(i, j, L):
    return (
        (i, (j + 1) % L),
        (i, (j - 1) % L),
        ((i + 1) % L, j),
        ((i - 1) % L, j),
    )


def xy_energy(state, J=1.0):
    L = state.shape[0]
    energy = 0.0
    for i in range(L):
        for j in range(L):
            energy -= J * np.cos(state[i, j] - state[i, (j + 1) % L])
            energy -= J * np.cos(state[i, j] - state[(i + 1) % L, j])
    return energy


def xy_magnetization(state):
    """Returns |m| = |<e^{i theta}>|, i.e. the modulus of the mean spin vector."""
    return float(np.abs(np.mean(np.exp(1j * state))))


def reflect_angles(angles, phi):
    """Reflect angles across the hyperplane perpendicular to r=(cos phi, sin phi).
    Derivation: for spin s=(cos theta, sin theta) the reflection gives
    theta' = 2*phi + pi - theta  (mod 2*pi).
    """
    return np.mod(2.0 * phi + np.pi - angles, 2.0 * np.pi)


def wolff_update_xy(state, beta, J=1.0):
    """Single Wolff cluster update for the 2-D XY model.

    Returns
    -------
    state        : updated spin configuration
    cluster_size : number of spins in the cluster
    proj_sum     : sum of cos(theta_i - phi) over cluster spins,
                   evaluated BEFORE reflection.  Required for the O(2)
                   improved susceptibility estimator.
    """
    L = state.shape[0]
    i0 = np.random.randint(L)
    j0 = np.random.randint(L)
    phi = np.random.uniform(0.0, 2.0 * np.pi)

    cluster = np.zeros((L, L), dtype=bool)
    cluster[i0, j0] = True
    stack = [(i0, j0)]

    while stack:
        i, j = stack.pop()
        for ni, nj in neighbor_indices(i, j, L):
            if cluster[ni, nj]:
                continue

            proj_i = np.cos(state[i, j] - phi)
            proj_j = np.cos(state[ni, nj] - phi)
            bond_prob = 1.0 - np.exp(-2.0 * beta * J * max(0.0, proj_i * proj_j))

            if np.random.rand() < bond_prob:
                cluster[ni, nj] = True
                stack.append((ni, nj))

    # Compute projection sum on the ORIGINAL angles, before flipping.
    proj_sum = float(np.sum(np.cos(state[cluster] - phi)))
    state[cluster] = reflect_angles(state[cluster], phi)
    return state, int(cluster.sum()), proj_sum


def equilibrate_xy_state(L, beta, n_eq=None, J=1.0):
    """Thermalise a fresh random state.

    The default n_eq is raised to 50*L^2 (minimum 2000 sweeps) to ensure
    adequate equilibration near the BKT transition at beta ~ 1.12.
    """
    if n_eq is None:
        n_eq = max(50 * L * L, 2000)

    state = random_state(L)
    for _ in range(n_eq):
        state, _, _ = wolff_update_xy(state, beta, J)
    return state


def measure_xy_observables(state, beta, J=1.0, n_samples=250):
    """Measure thermodynamic observables via n_samples Wolff sweeps.

    Susceptibility uses the standard estimator chi = beta * N * <|m|^2>.
    Note: we do NOT subtract <|m|>^2.  For a finite XY lattice <|m|> != 0
    even in the disordered phase (finite-size pseudo-magnetisation), so
    the variance Var(|m|) is not chi.  The correct expression is
    chi = beta/N * <|sum_i s_i|^2> = beta * N * <|m|^2>.
    """
    N = state.size
    energies = []
    magnetizations = []
    cluster_sizes = []
    imp_estimators = []   # (n/|C|) * (sum cos(theta_i - phi))^2, n=2 for XY

    for _ in range(n_samples):
        state, cluster_size, proj_sum = wolff_update_xy(state, beta, J)
        energies.append(xy_energy(state, J))
        magnetizations.append(xy_magnetization(state))
        cluster_sizes.append(cluster_size)
        if cluster_size > 0:
            imp_estimators.append((2.0 / cluster_size) * proj_sum ** 2)

    energies = np.asarray(energies)
    magnetizations = np.asarray(magnetizations)
    energy = float(np.mean(energies))
    specific_heat = float((beta ** 2 / N) * (np.mean(energies ** 2) - energy ** 2))

    # FIX: chi = beta * N * <|m|^2>, not beta * N * Var(|m|)
    susceptibility = float(beta * N * np.mean(magnetizations ** 2))

    return {
        "energy": energy,
        "specific_heat": specific_heat,
        "susceptibility": susceptibility,
        "mean_cluster_size": float(np.mean(cluster_sizes)),
    }


def run_simulation(betas, L_list=(8, 16), n_eq=None, n_samples=250, J=1.0):
    ensure_dir(SAVE_DIR)
    records = []

    for beta in betas:
        for L in L_list:
            state = equilibrate_xy_state(L, beta, n_eq=n_eq, J=J)
            results = measure_xy_observables(state, beta, J=J, n_samples=n_samples)
            records.append(
                {
                    "L": L,
                    "beta": beta,
                    "energy": results["energy"],
                    "specific_heat": results["specific_heat"],
                    "susceptibility": results["susceptibility"],
                    "mean_cluster_size": results["mean_cluster_size"],
                }
            )

    df = pd.DataFrame(records)
    df.to_csv(SAVE_DIR / "xy_wolff_measurements.csv", index=False)
    return df


if __name__ == "__main__":
    run_simulation([0.5, 0.8, 1.1])
