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

def reweight_moments(energies, magnetizations, L,
                     beta0, beta_values):
    V = L * L

    m = magnetizations / V

    absm = np.abs(m)

    m2 = m ** 2
    m4 = m ** 4

    results = []

    for beta in beta_values:

        delta = beta - beta0

        # logarithmic weights improve numerical stability
        logw = -delta * energies
        logw -= np.max(logw)

        w = np.exp(logw)

        Z = np.sum(w)

        mean_absm = np.sum(w * absm) / Z
        mean_m2 = np.sum(w * m2) / Z
        mean_m4 = np.sum(w * m4) / Z

        results.append(
            {
                "beta": beta,
                "absm": mean_absm,
                "m2": mean_m2,
                "m4": mean_m4,
            }
        )

    return pd.DataFrame(results)

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
    import matplotlib.pyplot as plt

    run_finite_size_scan(L_list=[16, 32], n_meas=20_000, rng_seed=42)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, L in zip(axes, [16, 32]):
        data = np.load(DATA_DIR / f"timeseries_L{L}.npz")
        energies, magnetizations = data["energies"], data["magnetizations"]

        half_width = 3.0 / L
        beta_rw = np.linspace(BETA_C - half_width, BETA_C + half_width, 200)
        df_rw = reweight_moments(energies, magnetizations, L, BETA_C, beta_rw)

        ax.plot(beta_rw, df_rw["absm"], c="C0", label=r"$\langle|m|\rangle$ (reweight)")
        ax.plot(beta_rw, df_rw["m2"],   c="C1", label=r"$\langle m^2\rangle$ (reweight)")
        ax.plot(beta_rw, df_rw["m4"],   c="C2", label=r"$\langle m^4\rangle$ (reweight)")

        V = L * L
        state = np.random.choice(np.array([-1, 1], dtype=np.int8), size=(L, L))
        first = True
        for offset in [-2.0, -0.8, 0.8, 2.0]:
            beta_d = BETA_C + offset / L
            e_d, m_d = run_wolff_chain(state, beta_d, max(30 * V, 2000), 10_000, 1)
            m_d = m_d / V
            # label only on first offset to avoid four duplicate legend entries
            lbl = "direct sim" if first else None
            ax.scatter(beta_d, np.mean(np.abs(m_d)), c="C0", marker="x", s=50, zorder=5, label=lbl)
            ax.scatter(beta_d, np.mean(m_d ** 2),    c="C1", marker="x", s=50, zorder=5)
            ax.scatter(beta_d, np.mean(m_d ** 4),    c="C2", marker="x", s=50, zorder=5)
            first = False

        ax.axvline(BETA_C, ls="--", color="grey", lw=0.8, label=r"$\beta_c$")
        ax.set_title(f"L={L}")
        ax.set_xlabel(r"$\beta$")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(DATA_DIR / "reweighting_magnetisation.png", dpi=150)
    plt.show()