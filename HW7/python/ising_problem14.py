# """
# Problem 14 – Statistical errors of histograms (2D Ising, 16×16 at beta_c)
# =========================================================================

# Simulation details (as specified):
#   L        = 16
#   beta_c   = ln(1 + sqrt(2)) / 2  ≈ 0.440686
#   Algorithm: Metropolis single-spin-flip
#   Thermalisation: 5 000 sweeps
#   Measurement  : 2^16 = 65 536 sweeps
#   Blocks       : 16  (each block = 65536/16 = 4096 sweeps)

# Quantities measured:
#   1. Mean energy <E>/N and its statistical error (blocking).
#   2. Energy histogram H(E) as an unnormalized number histogram.
#   3. Integrated autocorrelation time tau_int,E from the full time series.
#   4. Comparison of measured sigma_H(E) vs sqrt(H(E)) for uncorrelated data,
#      and check whether sigma_H ≈ sqrt(2 tau_int) * sqrt(H(E)).
# """

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

L = 16
N = L * L
BETA_C = np.log(1.0 + np.sqrt(2.0)) / 2.0  # ≈ 0.440686
N_THERM = 5_000          # thermalisation sweeps
N_MEAS = 2**16           # = 65 536 measurement sweeps
N_BLOCKS = 16
BLOCK_SIZE = N_MEAS // N_BLOCKS   # = 4 096 sweeps per block


# ─────────────────────────────────────────────────────────────────────────────
# Metropolis update (pure NumPy, vectorised checkerboard)
# ─────────────────────────────────────────────────────────────────────────────


def total_energy(state):
    """Ising energy E = -J sum_{<ij>} s_i s_j, J=1."""
    return -float(
        np.sum(state * np.roll(state, 1, axis=0))
        + np.sum(state * np.roll(state, 1, axis=1))
    )


def metropolis_sweep(state, beta):
    """One sweep = N single-spin-flip Metropolis steps (random order)."""
    for _ in range(N):
        i = np.random.randint(0, L)
        j = np.random.randint(0, L)
        # sum of four neighbours
        nb = (
            state[(i - 1) % L, j]
            + state[(i + 1) % L, j]
            + state[i, (j - 1) % L]
            + state[i, (j + 1) % L]
        )
        delta_E = 2.0 * state[i, j] * nb
        if delta_E <= 0.0 or np.random.random() < np.exp(-beta * delta_E):
            state[i, j] = -state[i, j]
    return state


# ─────────────────────────────────────────────────────────────────────────────
# Integrated autocorrelation time
# ─────────────────────────────────────────────────────────────────────────────


def integrated_autocorrelation(ts, c_factor=6.0):
    """Estimate tau_int for a 1-D time series using the automatic windowing
    procedure of Madras & Sokal (1988).

    Parameters
    ----------
    ts       : 1-D array of measurements
    c_factor : window = min{t : Gamma(t) < 0 or t >= c_factor * tau_int(t)}

    Returns
    -------
    tau_int : integrated autocorrelation time
    Gamma   : normalised autocorrelation function (lag 0 ... window)
    window  : chosen window
    """
    n = len(ts)
    ts = np.asarray(ts, dtype=float)
    ts -= ts.mean()
    var0 = np.dot(ts, ts) / n
    if var0 == 0.0:
        return 0.0, np.array([1.0]), 0

    max_lag = n // 2
    Gamma = np.zeros(max_lag)
    Gamma[0] = 1.0

    tau = 0.5
    window = max_lag - 1

    for t in range(1, max_lag):
        c = np.dot(ts[: n - t], ts[t:]) / ((n - t) * var0)
        Gamma[t] = c
        tau += c
        if t >= c_factor * tau:
            window = t
            break

    return float(tau), Gamma[:window + 1], window


# ─────────────────────────────────────────────────────────────────────────────
# Main simulation
# ─────────────────────────────────────────────────────────────────────────────


def run_simulation(rng_seed=None):
    if rng_seed is not None:
        np.random.seed(rng_seed)

    # ── initialise ──────────────────────────────────────────────────────────
    state = np.random.choice([-1, 1], size=(L, L)).astype(np.int8)

    print(f"beta_c = {BETA_C:.6f}")
    print(f"Thermalising ({N_THERM} sweeps)...")
    t0 = time.time()
    for _ in range(N_THERM):
        metropolis_sweep(state, BETA_C)
    print(f"  done in {time.time()-t0:.1f}s")

    # ── measure ─────────────────────────────────────────────────────────────
    energy_series = np.empty(N_MEAS)

    print(f"Measuring ({N_MEAS} sweeps)...")
    t0 = time.time()
    for s in range(N_MEAS):
        metropolis_sweep(state, BETA_C)
        energy_series[s] = total_energy(state)
    print(f"  done in {time.time()-t0:.1f}s")

    np.save(DATA_DIR / "ising_energy_series.npy", energy_series)

    # ── mean energy ──────────────────────────────────────────────────────────
    mean_E = float(np.mean(energy_series))
    mean_e = mean_E / N   # per site

    # ── autocorrelation time ─────────────────────────────────────────────────
    tau_int, Gamma, window = integrated_autocorrelation(energy_series)
    print(f"\ntau_int,E = {tau_int:.2f}  (window={window})")

    # naive standard error and corrected (autocorrelation) standard error
    sigma_naive = float(np.std(energy_series, ddof=1) / np.sqrt(N_MEAS))
    sigma_auto  = sigma_naive * np.sqrt(2.0 * tau_int)
    print(f"<E>/N = {mean_e:.5f}")
    print(f"  sigma_naive = {sigma_naive:.5f}")
    print(f"  sigma_auto  = {sigma_auto:.5f}  (= sigma_naive * sqrt(2*tau))")

    # ── blocking analysis for energy ─────────────────────────────────────────
    block_means = np.array(
        [np.mean(energy_series[b * BLOCK_SIZE : (b + 1) * BLOCK_SIZE])
         for b in range(N_BLOCKS)]
    )
    sigma_block = float(np.std(block_means, ddof=1) / np.sqrt(N_BLOCKS))
    print(f"  sigma_block = {sigma_block:.5f}  ({N_BLOCKS} blocks of {BLOCK_SIZE})")

    # ── histogram ────────────────────────────────────────────────────────────
    # Ising energies are integers: E ranges from -2N to +2N in steps of 4
    E_arr = energy_series.astype(int)
    E_min, E_max = int(E_arr.min()), int(E_arr.max())

    # energy values that actually appear
    E_vals = np.arange(E_min, E_max + 4, 4)

    # full histogram (all N_MEAS sweeps)
    H_full = np.zeros(len(E_vals), dtype=int)
    for v, e in enumerate(E_vals):
        H_full[v] = int(np.sum(E_arr == e))

    # per-block histograms for error estimation
    H_blocks = np.zeros((N_BLOCKS, len(E_vals)), dtype=float)
    for b in range(N_BLOCKS):
        seg = E_arr[b * BLOCK_SIZE : (b + 1) * BLOCK_SIZE]
        for v, e in enumerate(E_vals):
            H_blocks[b, v] = float(np.sum(seg == e))

    # scale block histograms to full-run equivalent
    H_blocks_scaled = H_blocks * N_BLOCKS   # each block × 16 → same scale as full

    # measured statistical error from blocks
    sigma_H_block = np.std(H_blocks_scaled, axis=0, ddof=1) / np.sqrt(N_BLOCKS)

    # Poisson / uncorrelated reference: sqrt(H)
    sigma_H_poisson = np.sqrt(H_full.astype(float))

    # predicted correlated error: sqrt(2 tau_int) * sqrt(H)
    sigma_H_predicted = np.sqrt(2.0 * tau_int) * sigma_H_poisson

    # ratio measured / poisson
    ratio = np.where(
        sigma_H_poisson > 0,
        sigma_H_block / sigma_H_poisson,
        np.nan,
    )

    # ── save results ─────────────────────────────────────────────────────────
    df_hist = pd.DataFrame(
        {
            "E": E_vals,
            "H": H_full,
            "sigma_H_block": sigma_H_block,
            "sigma_H_poisson": sigma_H_poisson,
            "sigma_H_predicted": sigma_H_predicted,
            "ratio_block_over_poisson": ratio,
        }
    )
    df_hist.to_csv(DATA_DIR / "ising_histogram.csv", index=False)

    # scalar summary
    df_scalar = pd.DataFrame(
        [
            {
                "L": L,
                "beta_c": BETA_C,
                "N_meas": N_MEAS,
                "N_blocks": N_BLOCKS,
                "block_size": BLOCK_SIZE,
                "mean_E_per_site": mean_e,
                "sigma_naive": sigma_naive,
                "sigma_auto": sigma_auto,
                "sigma_block": sigma_block,
                "tau_int_E": tau_int,
                "sqrt_2tau": float(np.sqrt(2.0 * tau_int)),
            }
        ]
    )
    df_scalar.to_csv(DATA_DIR / "ising_problem14_summary.csv", index=False)

    # autocorrelation function
    lag_vals = np.arange(len(Gamma))
    df_acf = pd.DataFrame({"lag": lag_vals, "Gamma": Gamma})
    df_acf.to_csv(DATA_DIR / "ising_acf_energy.csv", index=False)

    print("\nHistogram error comparison (selected energies):")
    interesting = df_hist[df_hist["H"] > 5].head(10)
    print(interesting[["E", "H", "sigma_H_block", "sigma_H_poisson", "ratio_block_over_poisson"]].to_string(index=False))

    print(f"\nsqrt(2*tau_int) = {np.sqrt(2*tau_int):.3f}")
    print("Expected: ratio_block_over_poisson ≈ sqrt(2*tau_int)")

    return df_hist, df_scalar, df_acf


if __name__ == "__main__":
    run_simulation(rng_seed=42)
