# """
# Problem 13 – Cluster estimator for the 2D Ising correlation function
# =====================================================================

# Implements two estimators for G(R) = <s_0 s_R> along the x-axis:

#   Standard : G_std(R)  = mean of s_{x,y} * s_{x+R, y}  (all translations)
#   Cluster  : G_cls(R)  = (V/|C|) * Theta_C(0) * Theta_C(R)  (averaged over
#                           all translations of the pair that share the same
#                           vectorial distance R along x)

# Correlation length xi is extracted by fitting G(R) ~ A * exp(-R/xi) for
# large R using log-linear least-squares.

# Runs L=64 and L=128 at beta = 0.42, 0.40, 0.35, 0.30.
# """

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Ising single-cluster (Wolff) update
# ─────────────────────────────────────────────────────────────────────────────


@njit(cache=True)
def wolff_ising(state, beta):
    """Single Wolff cluster update for 2-D Ising on an L×L lattice.

    Returns
    -------
    cluster_mask : uint8 array (L,L), 1 inside cluster, 0 outside
    seed_i, seed_j : coordinates of the seed spin
    """
    L = state.shape[0]
    p_add = 1.0 - np.exp(-2.0 * beta)  # bond probability

    i0 = np.random.randint(0, L)
    j0 = np.random.randint(0, L)
    spin0 = state[i0, j0]

    cluster = np.zeros((L, L), dtype=np.uint8)
    cluster[i0, j0] = 1

    # explicit stack
    max_sites = L * L
    stack_i = np.empty(max_sites, dtype=np.int32)
    stack_j = np.empty(max_sites, dtype=np.int32)
    ptr = 0
    stack_i[0] = i0
    stack_j[0] = j0

    while ptr >= 0:
        ci = stack_i[ptr]
        cj = stack_j[ptr]
        ptr -= 1

        # four neighbours (periodic)
        for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ni = (ci + di) % L
            nj = (cj + dj) % L
            if cluster[ni, nj] == 0 and state[ni, nj] == spin0:
                if np.random.random() < p_add:
                    cluster[ni, nj] = 1
                    ptr += 1
                    stack_i[ptr] = ni
                    stack_j[ptr] = nj

    # flip cluster
    for i in range(L):
        for j in range(L):
            if cluster[i, j]:
                state[i, j] = -state[i, j]

    return cluster, i0, j0


@njit(cache=True)
def equilibrate_ising(state, beta, n_eq):
    for _ in range(n_eq):
        wolff_ising(state, beta)
    return state


# ─────────────────────────────────────────────────────────────────────────────
# Measurement of G(R) for R along x-axis
# ─────────────────────────────────────────────────────────────────────────────


@njit(cache=True)
def measure_standard(state):
    """G_std(R) = (1/N) sum_{x,y} s_{x,y} * s_{x+R, y}  for R=0..L/2."""
    L = state.shape[0]
    N = L * L
    half = L // 2
    G = np.zeros(half + 1)
    for R in range(half + 1):
        acc = 0.0
        for x in range(L):
            xR = (x + R) % L
            for y in range(L):
                acc += state[x, y] * state[xR, y]
        G[R] = acc / N
    return G


@njit(cache=True)
def measure_cluster(cluster, L):
    """G_cls(R) = (V/|C|) * Theta_C(x,y) * Theta_C(x+R, y),
    averaged over all (x,y) translations for each R.
    """
    N = L * L
    C_size = 0
    for i in range(L):
        for j in range(L):
            C_size += cluster[i, j]

    if C_size == 0:
        return np.zeros(L // 2 + 1)

    half = L // 2
    G = np.zeros(half + 1)
    prefactor = N / C_size

    for R in range(half + 1):
        acc = 0.0
        for x in range(L):
            xR = (x + R) % L
            for y in range(L):
                acc += cluster[x, y] * cluster[xR, y]
        G[R] = prefactor * (acc / N)

    return G


# ─────────────────────────────────────────────────────────────────────────────
# Fit G(R) ~ A exp(-R/xi) for R >= R_min
# ─────────────────────────────────────────────────────────────────────────────


def fit_xi(R_vals, G_vals, R_min=3):
    """Log-linear fit ln G = ln A - R/xi."""
    mask = (R_vals >= R_min) & (G_vals > 0)
    if mask.sum() < 3:
        return np.nan, np.nan
    R_fit = R_vals[mask].astype(float)
    lnG = np.log(G_vals[mask])
    # linear regression: lnG = a + b*R
    X = np.column_stack([np.ones_like(R_fit), R_fit])
    coeffs, residuals, rank, sv = np.linalg.lstsq(X, lnG, rcond=None)
    b = coeffs[1]
    if b >= 0:
        return np.nan, np.nan
    xi = -1.0 / b
    A = np.exp(coeffs[0])
    return xi, A


# ─────────────────────────────────────────────────────────────────────────────
# Main simulation loop
# ─────────────────────────────────────────────────────────────────────────────


def run_correlation(
    betas=(0.42, 0.40, 0.35, 0.30),
    L_list=(64, 128),
    n_eq_factor=20,
    n_samples=500,
    rng_seed=None,
):
    if rng_seed is not None:
        np.random.seed(rng_seed)

    all_G_records = []   # per-(L,beta,R) records for G(R)
    xi_records = []       # per-(L,beta) xi estimates

    for L in L_list:
        N = L * L
        n_eq = max(n_eq_factor * N, 2000)
        half = L // 2

        for beta in betas:
            t0 = time.time()
            print(f"  L={L:3d}  beta={beta:.2f}  n_eq={n_eq}  n_samples={n_samples}")

            state = np.random.choice([-1, 1], size=(L, L)).astype(np.int8)
            state = equilibrate_ising(state, beta, n_eq)

            G_std_acc = np.zeros(half + 1)
            G_cls_acc = np.zeros(half + 1)

            for s in range(n_samples):
                cluster, i0, j0 = wolff_ising(state, beta)
                G_std_acc += measure_standard(state)
                G_cls_acc += measure_cluster(cluster, L)

            G_std = G_std_acc / n_samples
            G_cls = G_cls_acc / n_samples

            R_vals = np.arange(half + 1)

            xi_std, A_std = fit_xi(R_vals, G_std)
            xi_cls, A_cls = fit_xi(R_vals, G_cls)

            xi_records.append(
                {
                    "L": L,
                    "beta": beta,
                    "xi_standard": xi_std,
                    "xi_cluster": xi_cls,
                    "A_standard": A_std,
                    "A_cluster": A_cls,
                }
            )

            for R in range(half + 1):
                all_G_records.append(
                    {
                        "L": L,
                        "beta": beta,
                        "R": R,
                        "G_standard": G_std[R],
                        "G_cluster": G_cls[R],
                    }
                )

            print(
                f"    xi_std={xi_std:.3f}  xi_cls={xi_cls:.3f}"
                f"  elapsed={time.time()-t0:.1f}s"
            )

    df_G = pd.DataFrame(all_G_records)
    df_xi = pd.DataFrame(xi_records)

    df_G.to_csv(DATA_DIR / "ising_correlation_G.csv", index=False)
    df_xi.to_csv(DATA_DIR / "ising_correlation_xi.csv", index=False)

    return df_G, df_xi


if __name__ == "__main__":
    df_G, df_xi = run_correlation(rng_seed=42)
    print("\nCorrelation lengths:")
    print(df_xi.to_string(index=False))
