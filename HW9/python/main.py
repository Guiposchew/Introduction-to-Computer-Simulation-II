from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "python" / "data"
FIG_DIR = ROOT / "tex" / "figures"
DATA_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

BETA_C = np.log(1.0 + np.sqrt(2.0)) / 2.0
RNG_SEED = 20260615


def get_random_state(L, rng):
    return rng.choice([-1, 1], size=(L, L)).astype(np.int8)


def get_energy(state):
    right = np.roll(state, -1, axis=1)
    down = np.roll(state, -1, axis=0)
    return float(-np.sum(state * (right + down)))


def get_magnetization_per_spin(state):
    return float(np.mean(state))


def wolff_update(state, beta, rng, J=1.0):
    L = state.shape[0]
    p_add = 1.0 - np.exp(-2.0 * beta * J)

    i0 = rng.integers(0, L)
    j0 = rng.integers(0, L)
    cluster_spin = state[i0, j0]

    stack = [(i0, j0)]
    in_cluster = np.zeros((L, L), dtype=bool)
    in_cluster[i0, j0] = True
    cluster_sites = []

    while stack:
        i, j = stack.pop()
        cluster_sites.append((i, j))
        for ni, nj in [
            ((i + 1) % L, j),
            ((i - 1) % L, j),
            (i, (j + 1) % L),
            (i, (j - 1) % L),
        ]:
            if not in_cluster[ni, nj] and state[ni, nj] == cluster_spin:
                if rng.random() < p_add:
                    in_cluster[ni, nj] = True
                    stack.append((ni, nj))

    for i, j in cluster_sites:
        state[i, j] *= -1

    return state, len(cluster_sites)


def wolff_run(state, beta, n_thermalization, n_measurements, rng, J=1.0):
    state = state.copy()

    for _ in range(n_thermalization):
        state, _ = wolff_update(state, beta, rng, J=J)

    energies = np.zeros(n_measurements, dtype=float)
    magnetizations = np.zeros(n_measurements, dtype=float)

    for m in range(n_measurements):
        state, _ = wolff_update(state, beta, rng, J=J)
        energies[m] = get_energy(state)
        magnetizations[m] = get_magnetization_per_spin(state)

    return energies, magnetizations


def wolff_magnetization_scan(state, beta, n_thermalization, n_measurements, rng, J=1.0):
    state = state.copy()
    for _ in range(n_thermalization):
        state, _ = wolff_update(state, beta, rng, J=J)

    magnetizations = np.zeros(n_measurements, dtype=float)
    for m in range(n_measurements):
        state, _ = wolff_update(state, beta, rng, J=J)
        magnetizations[m] = get_magnetization_per_spin(state)

    return magnetizations


def run_problem_17(L=16, betas=None, n_thermalization=300, n_measurements=1500):
    if betas is None:
        betas = [0.375, BETA_C, 0.475]

    all_energies = []
    rows = []
    for idx, beta in enumerate(betas):
        rng = np.random.default_rng(RNG_SEED + idx)
        state = get_random_state(L, rng)
        energies, _ = wolff_run(state, beta, n_thermalization, n_measurements, rng)
        all_energies.append(energies)
        rows.append(
            {
                "beta": float(beta),
                "L": int(L),
                "mean_energy": float(np.mean(energies)),
                "std_energy": float(np.std(energies, ddof=1)),
            }
        )

    df_direct = pd.DataFrame(rows)
    df_direct.to_csv(DATA_DIR / "hw9_problem17_direct.csv", index=False)

    e_min = int(np.min([np.min(vals) for vals in all_energies]) - 2)
    e_max = int(np.max([np.max(vals) for vals in all_energies]) + 2)
    bin_edges = np.arange(e_min, e_max + 4, 4)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    histograms = []
    sample_sizes = []
    for vals in all_energies:
        hist, _ = np.histogram(vals, bins=bin_edges)
        histograms.append(hist.astype(float))
        sample_sizes.append(len(vals))

    histograms = np.array(histograms)
    sample_sizes = np.array(sample_sizes)
    hist_sum = np.sum(histograms, axis=0)

    f = np.zeros(len(betas), dtype=float)
    for _ in range(400):
        p_of_e = np.zeros_like(centers, dtype=float)
        for e_idx, E in enumerate(centers):
            denom = 0.0
            for j, (n_j, beta_j, f_j) in enumerate(zip(sample_sizes, betas, f)):
                denom += n_j * np.exp(-beta_j * E + f_j)
            if denom > 0:
                p_of_e[e_idx] = hist_sum[e_idx] / denom

        new_f = np.zeros_like(f)
        for j, beta_j in enumerate(betas):
            z = np.sum(p_of_e * np.exp(-beta_j * centers))
            if z > 0:
                new_f[j] = -np.log(z)
            else:
                new_f[j] = f[j]

        if np.max(np.abs(new_f - f)) < 1e-10:
            f = new_f
            break
        f = new_f

    beta_grid = np.linspace(0.34, 0.49, 80)
    wham_energies = []
    for beta in beta_grid:
        weights = p_of_e * np.exp(-beta * centers)
        weights = weights / np.sum(weights)
        wham_energies.append(float(np.dot(centers, weights)))

    wham_df = pd.DataFrame(
        {
            "beta": beta_grid,
            "energy_wham": wham_energies,
        }
    )
    wham_df.to_csv(DATA_DIR / "hw9_problem17_wham.csv", index=False)

    plt.figure(figsize=(8, 4.5))
    for idx, (beta, hist) in enumerate(zip(betas, histograms)):
        plt.step(centers, hist / np.sum(hist), where="mid", label=rf"$\beta={beta:.3f}$")
    plt.xlabel("Energy")
    plt.ylabel("Normalized histogram")
    plt.title("Energy histograms used for WHAM")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "hw9_problem17_histograms.pdf")
    plt.close()

    plt.figure(figsize=(7, 4.5))
    plt.plot(df_direct["beta"], df_direct["mean_energy"], marker="o", label="direct sampling")
    plt.plot(beta_grid, wham_energies, color="C1", label="WHAM reweighting")
    plt.xlabel(r"$\beta$")
    plt.ylabel(r"$\langle E \rangle$")
    plt.title("Mean energy from direct sampling and WHAM")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "hw9_problem17_energy_vs_beta.pdf")
    plt.close()

    return df_direct, wham_df


def run_problem_18(Ls=(16, 32, 64, 128), n_thermalization=8, n_measurements=20):
    beta_min = 0.85 * BETA_C
    beta_max = 0.995 * BETA_C
    betas = np.linspace(beta_min, beta_max, 31)
    rows = []

    for L in Ls:
        for beta in betas:
            rng = np.random.default_rng(int(RNG_SEED + L * 1000 + beta * 100000))
            state = get_random_state(L, rng)
            magnetizations = wolff_magnetization_scan(
                state, beta, n_thermalization, n_measurements, rng
            )
            chi = beta * (L * L) * np.mean(np.array(magnetizations) ** 2)
            b = 1.0 - beta / BETA_C
            rows.append(
                {
                    "L": int(L),
                    "beta": float(beta),
                    "chi": float(chi),
                    "b": float(b),
                    "x": float(b * (L ** 1)),
                    "chi_scaled": float(chi * (L ** (-1.75))),
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(DATA_DIR / "hw9_problem18_susceptibility.csv", index=False)

    plt.figure(figsize=(7, 4.5))
    for L in Ls:
        sub = df[df["L"] == L]
        plt.plot(sub["beta"], sub["chi"], marker="o", label=rf"$L={L}$")
    plt.xlabel(r"$\beta$")
    plt.ylabel(r"$\chi$")
    plt.title(r"Susceptibility versus $\beta$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "hw9_problem18_chi_vs_beta.pdf")
    plt.close()

    plt.figure(figsize=(7, 4.5))
    for L in Ls:
        sub = df[df["L"] == L]
        b = sub["b"].to_numpy()
        chi = sub["chi"].to_numpy()
        plt.loglog(b, chi, marker="o", label=rf"$L={L}$")
    b_theory = np.geomspace(1e-4, 0.2, 200)
    plt.loglog(b_theory, 1.0 * b_theory ** (-1.75), "k--", label=r"$b^{-7/4}$")
    plt.xlabel(r"$b = 1 - \beta/\beta_c$")
    plt.ylabel(r"$\chi$")
    plt.title(r"Susceptibility in the critical region")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "hw9_problem18_loglog_vs_b.pdf")
    plt.close()

    plt.figure(figsize=(7, 4.5))
    for L in Ls:
        sub = df[df["L"] == L]
        x = sub["x"].to_numpy()
        chi_scaled = sub["chi_scaled"].to_numpy()
        plt.loglog(x, chi_scaled, marker="o", label=rf"$L={L}$")
    x_theory = np.geomspace(1e-2, 20, 200)
    plt.loglog(x_theory, x_theory ** (-1.75), "k--", label=r"$x^{-7/4}$")
    plt.xlabel(r"$x = (1 - \beta/\beta_c) L^{1/\nu}$")
    plt.ylabel(r"$\chi L^{-\gamma/\nu}$")
    plt.title("Master plot of the susceptibility")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "hw9_problem18_master_plot.pdf")
    plt.close()

    return df


def main():
    df_direct, wham_df = run_problem_17()
    df_susc = run_problem_18()

    print("Problem 17 complete.")
    print(df_direct.to_string(index=False))
    print(wham_df.head().to_string(index=False))
    print("Problem 18 complete.")
    print(df_susc.head().to_string(index=False))


if __name__ == "__main__":
    main()
