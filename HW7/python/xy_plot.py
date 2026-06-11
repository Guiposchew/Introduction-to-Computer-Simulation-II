from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"
FIG_DIR = Path(__file__).resolve().parents[1] / "tex" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def _col(df, name):
    """Safe column access that avoids crashes across versions."""
    if name in df.columns:
        return df[name]
    return None


def save_scan_plots():
    df = pd.read_csv(DATA_DIR / "xy_cluster_scan.csv")

    # Figure 1: Mean energy per site
    plt.figure(figsize=(8, 5))
    for L in sorted(df["L"].unique()):
        sub = df[df["L"] == L]
        plt.plot(sub["beta"], sub["energy_per_site"], marker="o", label=f"L={L}")
    plt.xlabel(r"$\beta$")
    plt.ylabel(r"$\langle e \rangle = \langle E \rangle / N$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "xy_energy_vs_beta1.pdf")
    plt.close()

    # Figure 2: Specific heat
    plt.figure(figsize=(8, 5))
    for L in sorted(df["L"].unique()):
        sub = df[df["L"] == L]
        plt.plot(sub["beta"], sub["specific_heat"], marker="o", label=f"L={L}")
    plt.xlabel(r"$\beta$")
    plt.ylabel("Specific heat")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "xy_specific_heat_vs_beta1.pdf")
    plt.close()

    # Figure 3: Improved susceptibility
    plt.figure(figsize=(8, 5))
    for L in sorted(df["L"].unique()):
        sub = df[df["L"] == L]
        plt.plot(
            sub["beta"],
            sub["susceptibility_improved"],
            marker="o",
            label=f"L={L}",
        )
    plt.xlabel(r"$\beta$")
    plt.ylabel(r"$\chi_{\mathrm{imp}}$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "xy_susceptibility_vs_beta1.pdf")
    plt.close()

    # Figure 4: Mean cluster size
    plt.figure(figsize=(8, 5))
    for L in sorted(df["L"].unique()):
        sub = df[df["L"] == L]
        plt.plot(sub["beta"], sub["mean_cluster_size"], marker="o", label=f"L={L}")
    plt.xlabel(r"$\beta$")
    plt.ylabel(r"$\langle |C| \rangle$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "xy_mean_cluster_size_vs_beta1.pdf")
    plt.close()


def save_high_temp_plots():
    df = pd.read_csv(DATA_DIR / "xy_high_temp_compare.csv")

    # safe fallback for old/new column name
    ratio = _col(df, "ratio_mean_cluster_to_chi_beta")
    if ratio is None:
        ratio = _col(df, "ratio_std_over_imp")

    # Figure 5: Standard vs improved susceptibility
    plt.figure(figsize=(6, 5))
    plt.plot(df["beta"], df["susceptibility_standard"], marker="o", label="standard")
    plt.plot(
        df["beta"],
        df["susceptibility_improved"],
        marker="x",
        linestyle="--",
        label="improved",
    )
    plt.xlabel(r"$\beta$")
    plt.ylabel(r"$\chi$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "xy_high_temp_susceptibility_comparison1.pdf")
    plt.close()

    # Figure 6: Mean cluster size
    plt.figure(figsize=(6, 5))
    plt.plot(df["beta"], df["mean_cluster_size"], marker="o")
    plt.xlabel(r"$\beta$")
    plt.ylabel(r"$\langle |C| \rangle$")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "xy_high_temp_cluster_size1.pdf")
    plt.close()

    # Figure 7: Ratio plot (OLD STYLE, no renaming assumptions)
    plt.figure(figsize=(6, 5))
    plt.plot(df["beta"], ratio, marker="o")
    plt.axhline(0.81, linestyle="--", color="gray", linewidth=0.8)
    plt.xlabel(r"$\beta$")
    plt.ylabel(r"$\langle |C| \rangle / (\chi / \beta)$")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "xy_high_temp_cluster_ratio1.pdf")
    plt.close()


if __name__ == "__main__":
    save_scan_plots()
    save_high_temp_plots()