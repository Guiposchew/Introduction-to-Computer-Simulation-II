from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

from main import neighbor_indices, random_state, reflect_angles

DATA_DIR = Path(__file__).resolve().parent / "data"
SCAN_FILE = DATA_DIR / "xy_cluster_scan.csv"
COMPARE_FILE = DATA_DIR / "xy_high_temp_compare.csv"


# ============================================================
# NUMBA-ACCELERATED LOW-LEVEL ROUTINES
# ============================================================


@njit(cache=True)
def _reflect_angles_numba(cluster_spins, phi):
    return 2.0 * phi + np.pi - cluster_spins


@njit(cache=True)
def energy_xy(state, J=1.0):
    """Total XY energy with periodic boundaries."""
    L = state.shape[0]
    E = 0.0

    for i in range(L):
        ip = (i + 1) % L
        for j in range(L):
            jp = (j + 1) % L

            theta = state[i, j]

            E -= J * np.cos(theta - state[i, jp])
            E -= J * np.cos(theta - state[ip, j])

    return E


@njit(cache=True)
def m2_xy(state):
    """Squared magnetization."""
    L = state.shape[0]
    N = L * L

    mx = 0.0
    my = 0.0

    for i in range(L):
        for j in range(L):
            theta = state[i, j]
            mx += np.cos(theta)
            my += np.sin(theta)

    mx /= N
    my /= N

    return mx * mx + my * my


@njit(cache=True)
def wolff_update_xy_info(state, beta, J=1.0):
    """Correct Wolff update for XY model with consistent improved estimator.

    Key fix:
    - cluster is built on ORIGINAL configuration
    - spins are NOT modified during cluster construction
    - proj_sum is computed from original spins only
    - flips applied only AFTER cluster is complete
    """

    L = state.shape[0]

    i0 = np.random.randint(0, L)
    j0 = np.random.randint(0, L)
    phi = np.random.uniform(0.0, 2.0 * np.pi)

    cluster = np.zeros((L, L), dtype=np.uint8)

    max_sites = L * L
    stack_i = np.empty(max_sites, dtype=np.int32)
    stack_j = np.empty(max_sites, dtype=np.int32)
    stack_ptr = 0

    cluster[i0, j0] = 1
    stack_i[0] = i0
    stack_j[0] = j0

    cluster_sites_i = np.empty(max_sites, dtype=np.int32)
    cluster_sites_j = np.empty(max_sites, dtype=np.int32)
    cluster_count = 0

    while stack_ptr >= 0:
        i = stack_i[stack_ptr]
        j = stack_j[stack_ptr]
        stack_ptr -= 1

        if cluster_sites_i is None:
            pass

        # store site
        cluster_sites_i[cluster_count] = i
        cluster_sites_j[cluster_count] = j
        cluster_count += 1

        theta_ij = state[i, j]
        proj_i = np.cos(theta_ij - phi)

        # neighbors
        ip = (i + 1) % L
        im = (i - 1 + L) % L
        jp = (j + 1) % L
        jm = (j - 1 + L) % L

        # right
        ni, nj = ip, j
        if cluster[ni, nj] == 0:
            theta_n = state[ni, nj]
            prod = proj_i * np.cos(theta_n - phi)
            if prod > 0.0:
                if np.random.random() < (1.0 - np.exp(-2.0 * beta * J * prod)):
                    cluster[ni, nj] = 1
                    stack_ptr += 1
                    stack_i[stack_ptr] = ni
                    stack_j[stack_ptr] = nj

        # left
        ni, nj = im, j
        if cluster[ni, nj] == 0:
            theta_n = state[ni, nj]
            prod = proj_i * np.cos(theta_n - phi)
            if prod > 0.0:
                if np.random.random() < (1.0 - np.exp(-2.0 * beta * J * prod)):
                    cluster[ni, nj] = 1
                    stack_ptr += 1
                    stack_i[stack_ptr] = ni
                    stack_j[stack_ptr] = nj

        # up
        ni, nj = i, jp
        if cluster[ni, nj] == 0:
            theta_n = state[ni, nj]
            prod = proj_i * np.cos(theta_n - phi)
            if prod > 0.0:
                if np.random.random() < (1.0 - np.exp(-2.0 * beta * J * prod)):
                    cluster[ni, nj] = 1
                    stack_ptr += 1
                    stack_i[stack_ptr] = ni
                    stack_j[stack_ptr] = nj

        # down
        ni, nj = i, jm
        if cluster[ni, nj] == 0:
            theta_n = state[ni, nj]
            prod = proj_i * np.cos(theta_n - phi)
            if prod > 0.0:
                if np.random.random() < (1.0 - np.exp(-2.0 * beta * J * prod)):
                    cluster[ni, nj] = 1
                    stack_ptr += 1
                    stack_i[stack_ptr] = ni
                    stack_j[stack_ptr] = nj

    # compute proj_sum from ORIGINAL configuration
    proj_sum = 0.0
    cluster_size = cluster_count

    for k in range(cluster_count):
        i = cluster_sites_i[k]
        j = cluster_sites_j[k]
        proj_sum += np.cos(state[i, j] - phi)

    # apply flip AFTER cluster construction
    for k in range(cluster_count):
        i = cluster_sites_i[k]
        j = cluster_sites_j[k]
        theta = state[i, j]
        state[i, j] = 2.0 * phi + np.pi - theta

    return state, cluster_size, proj_sum


@njit(cache=True)
def equilibrate_xy_state(state, beta, n_eq, J=1.0):
    for _ in range(n_eq):
        state, _, _ = wolff_update_xy_info(state, beta, J)
    return state


# ============================================================
# HIGH-LEVEL DATA COLLECTION
# ============================================================


def run_beta_scan(betas, L_list=(8, 24), n_samples=200, J=1.0):
    """Scan over beta collecting thermodynamic observables."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    records = []

    for L in L_list:
        N = L * L

        # Strong equilibration near BKT
        n_eq = max(50 * N, 2000)

        for beta in betas:
            print(f"Running L={L}, beta={beta:.3f}")

            state = random_state(L).astype(np.float64)

            # JIT warmup on first call
            state, _, _ = wolff_update_xy_info(state, beta, J)

            state = equilibrate_xy_state(state, beta, n_eq, J)

            energies = np.empty(n_samples, dtype=np.float64)
            m2_values = np.empty(n_samples, dtype=np.float64)
            cluster_sizes = np.empty(n_samples, dtype=np.float64)
            imp_values = np.empty(n_samples, dtype=np.float64)

            for s in range(n_samples):
                state, cluster_size, proj_sum = wolff_update_xy_info(
                    state,
                    beta,
                    J,
                )

                energies[s] = energy_xy(state, J)
                m2_values[s] = m2_xy(state)
                cluster_sizes[s] = cluster_size

                if cluster_size > 0:
                    imp_values[s] = (
                        2.0 / cluster_size
                    ) * proj_sum * proj_sum
                else:
                    imp_values[s] = 0.0

            mean_energy = np.mean(energies)

            chi_imp = float(beta * np.mean(imp_values))
            chi_std = float(beta * N * np.mean(m2_values))

            record = {
                "L": L,
                "beta": beta,
                "energy": float(mean_energy),
                "energy_per_site": float(mean_energy / N),
                "specific_heat": float(
                    (beta ** 2 / N)
                    * (
                        np.mean(energies ** 2)
                        - mean_energy ** 2
                    )
                ),
                "susceptibility_standard": chi_std,
                "susceptibility_improved": chi_imp,
                "mean_cluster_size": float(np.mean(cluster_sizes)),
                "ratio_mean_cluster_to_chi_beta": (
                    float(np.mean(cluster_sizes) / (chi_imp / beta))
                    if chi_imp > 0 else float("nan")
                ),
            }

            records.append(record)

    df = pd.DataFrame(records)
    df.to_csv(SCAN_FILE, index=False)

    return df



def run_high_temperature_compare(
    betas=(1.0, 0.9, 0.8, 0.7, 0.6, 0.5),
    L=24,
    n_samples=1000,
    J=1.0,
):
    """Compare standard and improved susceptibility estimators."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    N = L * L
    n_eq = max(50 * N, 2000)

    records = []

    for beta in betas:
        print(f"High-T comparison beta={beta:.3f}")

        state = random_state(L).astype(np.float64)

        # JIT warmup
        state, _, _ = wolff_update_xy_info(state, beta, J)

        state = equilibrate_xy_state(state, beta, n_eq, J)

        m2_values = np.empty(n_samples, dtype=np.float64)
        cluster_sizes = np.empty(n_samples, dtype=np.float64)
        imp_values = np.empty(n_samples, dtype=np.float64)

        for s in range(n_samples):
            state, cluster_size, proj_sum = wolff_update_xy_info(
                state,
                beta,
                J,
            )

            m2_values[s] = m2_xy(state)
            cluster_sizes[s] = cluster_size

            if cluster_size > 0:
                imp_values[s] = (
                    2.0 / cluster_size
                ) * proj_sum * proj_sum
            else:
                imp_values[s] = 0.0

        chi_imp = float(beta * np.mean(imp_values))
        chi_std = float(beta * N * np.mean(m2_values))

        ratio = (
            float(np.mean(cluster_sizes) / (chi_imp / beta))
            if chi_imp > 0 else float("nan")
        )

        records.append(
            {
                "L": L,
                "beta": beta,
                "susceptibility_standard": chi_std,
                "susceptibility_improved": chi_imp,
                "mean_cluster_size": float(np.mean(cluster_sizes)),
                "ratio_mean_cluster_to_chi_beta": ratio,
            }
        )

    df = pd.DataFrame(records)
    df.to_csv(COMPARE_FILE, index=False)

    return df


# ============================================================
# MAIN
# ============================================================


def main():
    scan_df = run_beta_scan(
        np.arange(0.0, 2.05, 0.05),
        L_list=(8, 24),
    )

    compare_df = run_high_temperature_compare()

    print(scan_df.head())
    print(compare_df)


if __name__ == "__main__":
    main()
