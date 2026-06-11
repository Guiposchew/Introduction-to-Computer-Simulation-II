# """
# Plotting for Problems 13 and 14.

# Problem 13 plots:
#   - G(R) standard vs cluster estimator (log scale) per (L, beta)
#   - xi vs beta for L=64 and L=128

# Problem 14 plots:
#   - Energy time series and autocorrelation function
#   - Histogram H(E) with error bars: measured (blocks) vs sqrt(H)
#   - Ratio sigma_H / sqrt(H) vs E with horizontal line at sqrt(2*tau_int)
# """

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"
FIG_DIR = Path(__file__).resolve().parents[1] / "tex" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Problem 13
# ─────────────────────────────────────────────────────────────────────────────


def plot_correlation_functions():
    df = pd.read_csv(DATA_DIR / "ising_correlation_G.csv")
    df_xi = pd.read_csv(DATA_DIR / "ising_correlation_xi.csv")

    betas = sorted(df["beta"].unique())
    L_list = sorted(df["L"].unique())

    cmap = plt.get_cmap("tab10")
    beta_colors = {b: cmap(i) for i, b in enumerate(betas)}

    # ── Fig 1: G(R) for each L, both estimators overlaid ────────────────────
    for L in L_list:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
        fig.suptitle(rf"Correlation function $G(R)$ along $x$-axis, $L={L}$")

        for ax, estimator, col_name, label in zip(
            axes,
            ("standard", "cluster"),
            ("G_standard", "G_cluster"),
            ("Standard estimator", "Cluster estimator"),
        ):
            for beta in betas:
                sub = df[(df["L"] == L) & (df["beta"] == beta)].sort_values("R")
                R = sub["R"].values
                G = sub[col_name].values
                mask = G > 0
                ax.semilogy(
                    R[mask],
                    G[mask],
                    marker="o",
                    markersize=3,
                    color=beta_colors[beta],
                    label=rf"$\beta={beta}$",
                )

                # overlay fit line
                row_xi = df_xi[(df_xi["L"] == L) & (df_xi["beta"] == beta)]
                A_col = "A_standard" if estimator == "standard" else "A_cluster"
                xi_col = "xi_standard" if estimator == "standard" else "xi_cluster"
                if len(row_xi) > 0:
                    xi = float(row_xi[xi_col].values[0])
                    A  = float(row_xi[A_col].values[0])
                    if not (np.isnan(xi) or np.isnan(A)):
                        R_fit = np.linspace(3, R[mask].max(), 100)
                        ax.semilogy(
                            R_fit,
                            A * np.exp(-R_fit / xi),
                            linestyle="--",
                            color=beta_colors[beta],
                            alpha=0.5,
                            linewidth=1.0,
                        )

            ax.set_xlabel(r"$R$")
            ax.set_ylabel(r"$G(R)$")
            ax.set_title(label)
            ax.legend(fontsize=8)
            ax.grid(True, which="both", alpha=0.3)

        plt.tight_layout()
        plt.savefig(FIG_DIR / f"ising_correlation_G_L{L}.pdf")
        plt.close()
        print(f"Saved ising_correlation_G_L{L}.pdf")

    # ── Fig 2: xi vs beta ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    markers = {"xi_standard": "o", "xi_cluster": "s"}
    labels  = {"xi_standard": "standard", "xi_cluster": "cluster"}
    ls      = {"xi_standard": "-",  "xi_cluster": "--"}

    for L in L_list:
        sub = df_xi[df_xi["L"] == L].sort_values("beta")
        for col in ("xi_standard", "xi_cluster"):
            ax.plot(
                sub["beta"],
                sub[col],
                marker=markers[col],
                linestyle=ls[col],
                label=rf"$L={L}$, {labels[col]}",
            )

    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$\xi(\beta)$")
    ax.set_title("Correlation length vs. inverse temperature")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "ising_xi_vs_beta.pdf")
    plt.close()
    print("Saved ising_xi_vs_beta.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Problem 14
# ─────────────────────────────────────────────────────────────────────────────


def plot_problem14():
    df_hist    = pd.read_csv(DATA_DIR / "ising_histogram.csv")
    df_summary = pd.read_csv(DATA_DIR / "ising_problem14_summary.csv")
    df_acf     = pd.read_csv(DATA_DIR / "ising_acf_energy.csv")

    tau_int   = float(df_summary["tau_int_E"].iloc[0])
    sqrt_2tau = float(df_summary["sqrt_2tau"].iloc[0])
    mean_e    = float(df_summary["mean_E_per_site"].iloc[0])
    sigma_b   = float(df_summary["sigma_block"].iloc[0])
    sigma_a   = float(df_summary["sigma_auto"].iloc[0])

    # ── Fig 3: Autocorrelation function ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    lag = df_acf["lag"].values
    ax.plot(lag, df_acf["Gamma"].values, lw=1.5, color="steelblue")
    ax.axhline(0, color="black", lw=0.8, linestyle="--")
    ax.set_xlabel(r"$t$ (sweeps)")
    ax.set_ylabel(r"$\Gamma_E(t) / \Gamma_E(0)$")
    ax.set_title(
        rf"Energy autocorrelation, $\beta_c={0.440686:.6f}$, $L=16$"
        f"\n$\\tau_{{\\mathrm{{int}},E}} = {tau_int:.2f}$"
    )
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "ising_autocorrelation_E.pdf")
    plt.close()
    print("Saved ising_autocorrelation_E.pdf")

    # ── Fig 4: Histogram with error bars ─────────────────────────────────────
    # Only plot energies that actually appear
    mask = df_hist["H"] > 0
    sub  = df_hist[mask]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(
        sub["E"],
        sub["H"],
        width=3.0,
        color="lightsteelblue",
        edgecolor="steelblue",
        linewidth=0.5,
        label=r"$H(E)$",
        zorder=2,
    )
    ax.errorbar(
        sub["E"],
        sub["H"],
        yerr=sub["sigma_H_block"],
        fmt="none",
        ecolor="firebrick",
        elinewidth=1.2,
        capsize=3,
        label=r"$\sigma_H$ (blocks)",
        zorder=3,
    )
    ax.errorbar(
        sub["E"] + 1.5,
        sub["H"],
        yerr=sub["sigma_H_poisson"],
        fmt="none",
        ecolor="darkorange",
        elinewidth=1.0,
        capsize=3,
        alpha=0.7,
        label=r"$\sqrt{H(E)}$ (Poisson)",
        zorder=3,
    )

    ax.set_xlabel(r"$E$")
    ax.set_ylabel(r"$H(E)$")
    ax.set_title(
        rf"Energy histogram $H(E)$, $L=16$, $\beta_c$"
        f"\n16 blocks of 4096 sweeps"
    )
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "ising_histogram_E.pdf")
    plt.close()
    print("Saved ising_histogram_E.pdf")

    # ── Fig 5: Ratio sigma_H / sqrt(H) ──────────────────────────────────────
    sub2 = df_hist[df_hist["H"] > 10].copy()  # ignore sparsely-visited bins
    ratio = sub2["ratio_block_over_poisson"].values

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sub2["E"], ratio, marker="o", markersize=4, color="steelblue", label="measured ratio")
    ax.axhline(
        sqrt_2tau,
        linestyle="--",
        color="firebrick",
        label=rf"$\sqrt{{2\tau_{{int}}}} = {sqrt_2tau:.2f}$",
    )
    ax.axhline(1.0, linestyle=":", color="gray", linewidth=0.8, label="Poisson reference (1)")
    ax.set_xlabel(r"$E$")
    ax.set_ylabel(r"$\sigma_H(E) \,/\, \sqrt{H(E)}$")
    ax.set_title(
        rf"Ratio of measured to Poisson histogram error, $L=16$, $\beta_c$"
        f"\n$\\tau_{{\\mathrm{{int}},E}} = {tau_int:.2f}$"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "ising_histogram_ratio.pdf")
    plt.close()
    print("Saved ising_histogram_ratio.pdf")

    # ── Print summary ────────────────────────────────────────────────────────
    print("\nProblem 14 summary:")
    print(f"  <E>/N          = {mean_e:.5f}")
    print(f"  sigma_block    = {sigma_b:.5f}")
    print(f"  sigma_auto     = {sigma_a:.5f}")
    print(f"  tau_int,E      = {tau_int:.2f}")
    print(f"  sqrt(2*tau)    = {sqrt_2tau:.3f}")
    print(f"  mean ratio     = {np.nanmean(ratio):.3f}  (should ~ {sqrt_2tau:.3f})")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    plot_correlation_functions()
    plot_problem14()
