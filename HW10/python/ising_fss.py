"""
Problem 20 - Finite-size scaling in the 2D Ising model at the critical point
==============================================================================

Simulate the 2D Ising model EXACTLY AT beta_c = ln(1+sqrt(2))/2 using the
single-cluster (Wolff) algorithm, for L = 8, 16, 32, 64, 128 with periodic
boundary conditions.

From the time series of energy E and magnetization M, compute:
  - specific heat        C(L)   = beta^2/V * (<E^2> - <E>^2)
  - susceptibility        chi(L) = beta * V * <m^2>          (m = M/V)
  - Binder parameter       U4(L)  = 1 - <m^4>/(3<m^2>^2)
  - second Binder param    U2(L)  = 1 - <m^2>/(3<|m|>^2)
  - dU4/dbeta, dU2/dbeta, d ln<m^2>/dbeta, d ln<|m|>/dbeta
    using the exact identities derived in Problem 19:

      d<A>/dbeta = V*(<A*e> - <A><e>)                          (*)

      dU4/dbeta = V*(1-U4)*(<e> - 2<m^2 e>/<m^2> + <m^4 e>/<m^4>)
      dU2/dbeta = V*(1-U2)*(<e> - 2<|m| e>/<|m|> + <m^2 e>/<m^2>)
      d ln<m^2>/dbeta  = V*(<e> - <m^2 e>/<m^2>)
      d ln<|m|>/dbeta  = V*(<e> - <|m| e>/<|m|>)

All derivatives are computed DIRECTLY from the same Monte Carlo time series
used for the primary observables (no separate simulation needed), since they
are just different moment combinations of the same (e, m) samples.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

BETA_C = np.log(1.0 + np.sqrt(2.0)) / 2.0   # exact critical point


# ─────────────────────────────────────────────────────────────────────────────
# Wolff single-cluster update (numba-JIT)
# ─────────────────────────────────────────────────────────────────────────────

@njit(cache=True)
def wolff_step(state, beta):
    """Single Wolff cluster update. Returns nothing; updates state in place
    and returns the cluster size (not needed here but cheap to expose)."""
    L = state.shape[0]
    p_add = 1.0 - np.exp(-2.0 * beta)

    i0 = np.random.randint(0, L)
    j0 = np.random.randint(0, L)
    spin0 = state[i0, j0]

    visited = np.zeros((L, L), dtype=np.uint8)
    visited[i0, j0] = 1

    max_sites = L * L
    stack_i = np.empty(max_sites, dtype=np.int32)
    stack_j = np.empty(max_sites, dtype=np.int32)
    ptr = 0
    stack_i[0] = i0
    stack_j[0] = j0
    cluster_size = 1

    while ptr >= 0:
        ci = stack_i[ptr]
        cj = stack_j[ptr]
        ptr -= 1
        for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ni = (ci + di) % L
            nj = (cj + dj) % L
            if visited[ni, nj] == 0 and state[ni, nj] == spin0:
                if np.random.random() < p_add:
                    visited[ni, nj] = 1
                    cluster_size += 1
                    ptr += 1
                    stack_i[ptr] = ni
                    stack_j[ptr] = nj

    for i in range(L):
        for j in range(L):
            if visited[i, j]:
                state[i, j] = -state[i, j]

    return cluster_size


@njit(cache=True)
def total_energy(state):
    """E = -J * sum_{<ij>} s_i s_j  (J=1), periodic BC."""
    L = state.shape[0]
    E = 0
    for i in range(L):
        ip = (i + 1) % L
        for j in range(L):
            jp = (j + 1) % L
            E -= state[i, j] * state[ip, j]
            E -= state[i, j] * state[i, jp]
    return float(E)


@njit(cache=True)
def total_magnetization(state):
    return float(np.sum(state))


@njit(cache=True)
def run_wolff_chain(state, beta, n_therm, n_meas, meas_stride):
    """Thermalise, then collect (E, M) every `meas_stride` Wolff updates.

    meas_stride > 1 is used for large L to reduce the cost of computing
    E and M every single step.  Numba-compiled for speed.
    """
    L = state.shape[0]

    for _ in range(n_therm):
        wolff_step(state, beta)

    energies = np.empty(n_meas, dtype=np.float64)
    magnetizations = np.empty(n_meas, dtype=np.float64)

    for s in range(n_meas):
        for _ in range(meas_stride):
            wolff_step(state, beta)
        energies[s] = total_energy(state)
        magnetizations[s] = total_magnetization(state)

    return energies, magnetizations


# ─────────────────────────────────────────────────────────────────────────────
# Observable post-processing (moments + derivatives via identity (*))
# ─────────────────────────────────────────────────────────────────────────────

def compute_observables(energies, magnetizations, L, beta=BETA_C):
    """Compute C, chi, U4, U2, and their beta-derivatives from a single
    (E, M) time series, using the exact fluctuation identities.

    e = E/V  (energy density)
    m = M/V  (magnetization density)
    """
    V = L * L
    e = energies / V
    m = magnetizations / V
    m_abs = np.abs(m)
    m2 = m ** 2
    m4 = m ** 4

    # ── primary moments ───────────────────────────────────────────────────
    mean_e   = np.mean(e)
    mean_e2  = np.mean(e ** 2)
    mean_m2  = np.mean(m2)
    mean_m4  = np.mean(m4)
    mean_mabs = np.mean(m_abs)

    # ── specific heat & susceptibility ─────────────────────────────────────
    specific_heat = (beta ** 2) * V * (mean_e2 - mean_e ** 2)
    susceptibility = beta * V * mean_m2

    # ── Binder parameters ───────────────────────────────────────────────────
    U4 = 1.0 - mean_m4 / (3.0 * mean_m2 ** 2)
    U2 = 1.0 - mean_m2 / (3.0 * mean_mabs ** 2)

    # ── cross moments needed for derivatives (identity *) ───────────────────
    mean_m2_e  = np.mean(m2 * e)
    mean_m4_e  = np.mean(m4 * e)
    mean_mabs_e = np.mean(m_abs * e)

    # d ln<m^2>/dbeta  = V*(<e> - <m^2 e>/<m^2>)
    dlnm2_dbeta = V * (mean_e - mean_m2_e / mean_m2)

    # d ln<|m|>/dbeta  = V*(<e> - <|m| e>/<|m|>)
    dlnmabs_dbeta = V * (mean_e - mean_mabs_e / mean_mabs)

    # dU4/dbeta = V*(1-U4)*(<e> - 2<m^2 e>/<m^2> + <m^4 e>/<m^4>)
    dU4_dbeta = V * (1.0 - U4) * (
        mean_e - 2.0 * mean_m2_e / mean_m2 + mean_m4_e / mean_m4
    )

    # dU2/dbeta = V*(1-U2)*(<e> - 2<|m| e>/<|m|> + <m^2 e>/<m^2>)
    dU2_dbeta = V * (1.0 - U2) * (
        mean_e - 2.0 * mean_mabs_e / mean_mabs + mean_m2_e / mean_m2
    )

    return {
        "L": L,
        "beta": beta,
        "energy_per_site": mean_e,
        "specific_heat": specific_heat,
        "susceptibility": susceptibility,
        "U4": U4,
        "U2": U2,
        "mean_m2": mean_m2,
        "mean_m4": mean_m4,
        "mean_mabs": mean_mabs,
        "dU4_dbeta": dU4_dbeta,
        "dU2_dbeta": dU2_dbeta,
        "dlnm2_dbeta": dlnm2_dbeta,
        "dlnmabs_dbeta": dlnmabs_dbeta,
    }


def bootstrap_errors(energies, magnetizations, L, beta=BETA_C,
                      n_boot=200, block_size=None):
    """Estimate statistical errors on all observables via block bootstrap.

    block_size defaults to a value comparable to the autocorrelation time
    of the Wolff algorithm (which is short, O(1) cluster updates), so a
    modest block size suffices.
    """
    n = len(energies)
    if block_size is None:
        block_size = max(n // 50, 10)
    n_blocks = n // block_size

    # Build block-averaged series for bootstrap resampling
    e_blocks = energies[: n_blocks * block_size].reshape(n_blocks, block_size)
    m_blocks = magnetizations[: n_blocks * block_size].reshape(n_blocks, block_size)

    keys = None
    boot_vals = {}

    for _ in range(n_boot):
        idx = np.random.randint(0, n_blocks, size=n_blocks)
        e_resampled = e_blocks[idx].ravel()
        m_resampled = m_blocks[idx].ravel()
        obs = compute_observables(e_resampled, m_resampled, L, beta)
        if keys is None:
            keys = [k for k in obs if k not in ("L", "beta")]
            boot_vals = {k: [] for k in keys}
        for k in keys:
            boot_vals[k].append(obs[k])

    errors = {f"d{k}": float(np.std(boot_vals[k], ddof=1)) for k in keys}
    return errors


# ─────────────────────────────────────────────────────────────────────────────
# Main driver
# ─────────────────────────────────────────────────────────────────────────────

def run_finite_size_scan(
    L_list=(8, 16, 32, 64, 128),
    n_eq_factor=30,     # equilibration Wolff steps per site
    n_meas=20000,       # number of measurement samples
    meas_stride=1,      # Wolff updates between successive measurements
    n_boot=200,
    rng_seed=None,
):
    if rng_seed is not None:
        np.random.seed(rng_seed)

    records = []

    for L in L_list:
        V = L * L
        n_eq = max(n_eq_factor * V, 2000)

        # For larger L, increase measurement stride slightly to reduce
        # residual autocorrelation between samples (Wolff still has
        # short but nonzero tau, especially exactly at beta_c).
        stride = meas_stride if L <= 32 else 2

        t0 = time.time()
        print(f"L={L:4d}  n_eq={n_eq:7d}  n_meas={n_meas:6d}  stride={stride}")

        state = np.random.choice(np.array([-1, 1], dtype=np.int8), size=(L, L))

        # JIT warmup
        _ = run_wolff_chain(state.copy(), BETA_C, 1, 1, 1)

        energies, magnetizations = run_wolff_chain(
            state, BETA_C, n_eq, n_meas, stride
        )

        obs = compute_observables(energies, magnetizations, L, BETA_C)
        errs = bootstrap_errors(energies, magnetizations, L, BETA_C, n_boot=n_boot)

        record = {**obs, **errs}
        records.append(record)

        np.savez(
            DATA_DIR / f"timeseries_L{L}.npz",
            energies=energies,
            magnetizations=magnetizations,
        )

        print(
            f"  C={obs['specific_heat']:.3f}  chi={obs['susceptibility']:.3f}  "
            f"U4={obs['U4']:.4f}±{errs['dU4']:.4f}  "
            f"dU4/dbeta={obs['dU4_dbeta']:.2f}±{errs['ddU4_dbeta']:.2f}  "
            f"t={time.time()-t0:.1f}s"
        )

    df = pd.DataFrame(records)
    df.to_csv(DATA_DIR / "ising_fss_betac.csv", index=False)
    return df


if __name__ == "__main__":
    df = run_finite_size_scan(rng_seed=42)
    print()
    print("=" * 90)
    print("FINITE-SIZE SCALING RESULTS AT beta_c")
    print("=" * 90)
    cols = ["L", "specific_heat", "susceptibility", "U4", "U2",
            "dU4_dbeta", "dU2_dbeta", "dlnm2_dbeta", "dlnmabs_dbeta"]
    print(df[cols].to_string(index=False))
