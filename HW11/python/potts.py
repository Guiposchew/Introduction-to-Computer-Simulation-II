import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import njit
import os
import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# Numba-JIT Wolff + energy for Potts
# ─────────────────────────────────────────────────────────────────────────────

@njit(cache=True)
def wolff_step_potts(state, beta, q):
    L = state.shape[0]
    p_add = 1.0 - np.exp(-beta)

    i0 = np.random.randint(0, L)
    j0 = np.random.randint(0, L)
    spin0 = state[i0, j0]

    visited = np.zeros((L, L), dtype=np.uint8)
    visited[i0, j0] = 1

    stack_i = np.empty(L * L, dtype=np.int32)
    stack_j = np.empty(L * L, dtype=np.int32)
    ptr = 0
    stack_i[0] = i0
    stack_j[0] = j0

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
                    ptr += 1
                    stack_i[ptr] = ni
                    stack_j[ptr] = nj

    # flip to a random different spin
    new_spin = np.int8((spin0 + 1 + np.random.randint(0, q - 1)) % q)
    for i in range(L):
        for j in range(L):
            if visited[i, j]:
                state[i, j] = new_spin


@njit(cache=True)
def potts_energy_njit(state):
    L = state.shape[0]
    E = 0
    for i in range(L):
        ip = (i + 1) % L
        for j in range(L):
            jp = (j + 1) % L
            E -= (state[i, j] == state[ip, j])
            E -= (state[i, j] == state[i, jp])
    return float(E)


@njit(cache=True)
def run_potts_chain(state, beta, q, n_therm, n_meas):
    for _ in range(n_therm):
        wolff_step_potts(state, beta, q)
    E = np.empty(n_meas, dtype=np.float64)
    for s in range(n_meas):
        wolff_step_potts(state, beta, q)
        E[s] = potts_energy_njit(state)
    return E


# ─────────────────────────────────────────────────────────────────────────────
# Binning estimator — store bin_sizes and ratio for reuse in plot
# ─────────────────────────────────────────────────────────────────────────────

def tau_int_binning(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    var0 = np.var(x, ddof=1)

    bin_sizes, vars_binned = [], []
    b = 1
    while b <= n // 8:
        n_bins = n // b
        blocked = x[: n_bins * b].reshape(n_bins, b).mean(axis=1)
        vars_binned.append(np.var(blocked, ddof=1) / (n_bins - 1) * n_bins)
        bin_sizes.append(b)
        b *= 2

    ratio = np.array(vars_binned) / var0
    plateau_vals = ratio[len(ratio) * 2 // 3 :]
    tau_int = 0.5 * np.mean(plateau_vals) * bin_sizes[len(bin_sizes) * 2 // 3]

    return tau_int, np.array(bin_sizes), ratio

def tau_int_acf(x, window_factor=6):
    """
    Estimate tau_int from the normalized autocorrelation function.
    Cuts off the sum at the first t where C(t) < 0, or at window_factor * tau_int,
    whichever comes first (Madras-Sokal automatic windowing).
    """
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = len(x)

    # full ACF via FFT — O(n log n) instead of O(n^2)
    f = np.fft.rfft(x, n=2 * n)
    acf_full = np.fft.irfft(f * np.conj(f))[:n]
    acf_full /= acf_full[0]   # normalize so C(0) = 1

    # Madras-Sokal automatic window: stop at first t where
    # cumulative sum exceeds window_factor * current tau estimate
    tau = 0.5
    for t in range(1, n // 2):
        tau += acf_full[t]
        if t >= window_factor * tau:   # window condition
            break

    return tau, acf_full

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    q        = 3
    BETA_PC  = np.log(1.0 + np.sqrt(q))
    L_LIST   = [16, 32, 64]
    N_EQ_FACTOR = 50       # Wolff is fast; 50/site is plenty
    N_MEAS      = 5_000

    Ls = np.array(L_LIST)

    os.makedirs("data", exist_ok=True)

    # JIT warmup on tiny lattice so compilation doesn't charge to L=16
    _s = np.zeros((4, 4), dtype=np.int8)
    run_potts_chain(_s, BETA_PC, q, 1, 1)

    tau_results   = []
    binning_data  = {}   # store (bin_sizes, ratio) per L — reused in plot
    acf_data = []

    for L in L_LIST:
        V     = L * L
        n_eq  = N_EQ_FACTOR * V
        state = np.random.randint(0, q, size=(L, L), dtype=np.int8)

        E_series = run_potts_chain(state, BETA_PC, q, n_eq, N_MEAS)

        tau_bin, bin_sizes, ratio = tau_int_binning(E_series)
        tau_acf, acf             = tau_int_acf(E_series)

        tau_results.append({"L": L, "tau_bin": tau_bin, "tau_acf": tau_acf})
        binning_data[L] = (bin_sizes, ratio)
        acf_data.append(acf)

        print(f"L={L:4d}  tau_int(bin)={tau_bin:.2f}  tau_int(acf)={tau_acf:.2f}")

    # ── figure 1: binning plateaus ─────────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    for L in L_LIST:
        bin_sizes, ratio = binning_data[L]
        ax1.semilogx(bin_sizes, 0.5 * (ratio - 1) * bin_sizes,
                     marker="o", ms=3, label=f"L={L}")
    ax1.set_xlabel("bin size $b$")
    ax1.set_ylabel(r"$\tau_\mathrm{int}$ estimate")
    ax1.set_title("Binning plateaus")
    ax1.legend(fontsize=8)
    fig1.tight_layout()
    fig1.savefig("data/binning_plateaus.png", dpi=150)

    # ── figure 2: ACF curves + scaling comparison ──────────────────────────
    fig2, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    for r in tau_results:
        L   = r["L"]
        acf = acf_data[L]
        t_max = min(int(20 * r["tau_acf"]) + 1, 200)
        ax.plot(np.arange(t_max), acf[:t_max], label=f"L={L}")
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.set_xlabel("$t$ (Wolff steps)")
    ax.set_ylabel("$C(t)/C(0)$")
    ax.set_title("Normalized ACF")
    ax.legend(fontsize=8)

    ax = axes[1]
    taus_bin = np.array([r["tau_bin"] for r in tau_results])
    taus_acf = np.array([r["tau_acf"] for r in tau_results])
    for taus, label, marker in [
        (taus_bin, "binning", "o"),
        (taus_acf, "ACF",     "s"),
    ]:
        z_fit, log_a = np.polyfit(np.log(Ls), np.log(taus), 1)
        L_fit = np.logspace(np.log10(Ls.min()), np.log10(Ls.max()), 100)
        ax.loglog(Ls, taus, marker=marker, ms=6, label=f"{label} data")
        ax.loglog(L_fit, np.exp(log_a) * L_fit ** z_fit, "--",
                  label=f"{label} $z={z_fit:.2f}$")
    ax.set_xlabel("$L$")
    ax.set_ylabel(r"$\tau_\mathrm{int}$")
    ax.set_title(r"$\tau_\mathrm{int} \propto L^z$")
    ax.legend(fontsize=8)

    fig2.tight_layout()
    fig2.savefig("data/autocorrelation_potts.png", dpi=150)
    plt.show()


    # for L in tqdm(L_LIST):
    #     V     = L * L
    #     n_eq  = N_EQ_FACTOR * V
    #     state = np.random.randint(0, q, size=(L, L), dtype=np.int8)

    #     E_series = run_potts_chain(state, BETA_PC, q, n_eq, N_MEAS)

    #     tau_bin, bin_sizes, ratio = tau_int_binning(E_series)
    #     tau_acf, acf             = tau_int_acf(E_series)

    #     tau_results.append({"L": L, "tau_bin": tau_bin, "tau_acf": tau_acf})
    #     binning_data[L] = (bin_sizes, ratio)
    #     acf_data[L]     = acf

    #     print(f"L={L:4d}  tau_int(bin)={tau_bin:.2f}  tau_int(acf)={tau_acf:.2f}")

    # # ── plots — no re-simulation ───────────────────────────────────────────
    # df   = pd.DataFrame(tau_results)
    # Ls   = df["L"].values.astype(float)
    # taus = df["tau_int"].values

    # z_fit, log_a = np.polyfit(np.log(Ls), np.log(taus), 1)
    # L_fit = np.logspace(np.log10(Ls.min()), np.log10(Ls.max()), 100)

    # fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # ax = axes[0]
    # for L in L_LIST:
    #     bin_sizes, ratio = binning_data[L]
    #     ax.semilogx(bin_sizes, 0.5 * (ratio - 1) * bin_sizes,
    #                 marker="o", ms=3, label=f"L={L}")
    # ax.set_xlabel("bin size $b$")
    # ax.set_ylabel(r"$\tau_\mathrm{int}$ estimate")
    # ax.set_title("Binning plateaus")
    # ax.legend(fontsize=8)

    # ax = axes[1]
    # ax.loglog(Ls, taus, "o", ms=6, label="measured")
    # ax.loglog(L_fit, np.exp(log_a) * L_fit ** z_fit, "--",
    #           label=f"fit $z={z_fit:.2f}$")
    # ax.set_xlabel("$L$")
    # ax.set_ylabel(r"$\tau_\mathrm{int}$")
    # ax.set_title(r"$\tau_\mathrm{int} \propto L^z$")
    # ax.legend(fontsize=8)

    # # ── figure 1: binning plateaus ─────────────────────────────────────────
    # fig1, ax1 = plt.subplots(figsize=(6, 4))
    # for L in L_LIST:
    #     bin_sizes, ratio = binning_data[L]
    #     ax1.semilogx(bin_sizes, 0.5 * (ratio - 1) * bin_sizes,
    #                  marker="o", ms=3, label=f"L={L}")
    # ax1.set_xlabel("bin size $b$")
    # ax1.set_ylabel(r"$\tau_\mathrm{int}$ estimate")
    # ax1.set_title("Binning plateaus")
    # ax1.legend(fontsize=8)
    # fig1.tight_layout()
    # fig1.savefig("data/binning_plateaus.png", dpi=150)

    # # ── figure 2: ACF curves + scaling comparison ──────────────────────────
    # fig2, axes = plt.subplots(1, 2, figsize=(10, 4))

    # ax = axes[0]
    # for r in tau_results:
    #     L   = r["L"]
    #     acf = acf_data[L]
    #     t_max = min(int(20 * r["tau_acf"]) + 1, 200)
    #     ax.plot(np.arange(t_max), acf[:t_max], label=f"L={L}")
    # ax.axhline(0, color="grey", lw=0.8, ls="--")
    # ax.set_xlabel("$t$ (Wolff steps)")
    # ax.set_ylabel("$C(t)/C(0)$")
    # ax.set_title("Normalized ACF")
    # ax.legend(fontsize=8)

    # ax = axes[1]
    # taus_bin = np.array([r["tau_bin"] for r in tau_results])
    # taus_acf = np.array([r["tau_acf"] for r in tau_results])
    # for taus, label, marker in [
    #     (taus_bin, "binning", "o"),
    #     (taus_acf, "ACF",     "s"),
    # ]:
    #     z_fit, log_a = np.polyfit(np.log(Ls), np.log(taus), 1)
    #     L_fit = np.logspace(np.log10(Ls.min()), np.log10(Ls.max()), 100)
    #     ax.loglog(Ls, taus, marker=marker, ms=6, label=f"{label} data")
    #     ax.loglog(L_fit, np.exp(log_a) * L_fit ** z_fit, "--",
    #               label=f"{label} $z={z_fit:.2f}$")
    # ax.set_xlabel("$L$")
    # ax.set_ylabel(r"$\tau_\mathrm{int}$")
    # ax.set_title(r"$\tau_\mathrm{int} \propto L^z$")
    # ax.legend(fontsize=8)

    # fig2.tight_layout()
    # fig2.savefig("data/autocorrelation_potts.png", dpi=150)
    # plt.show()

    # plt.tight_layout()
    # plt.savefig("data/autocorrelation_potts.png", dpi=150)
    # plt.show()