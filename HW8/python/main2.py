from pathlib import Path
import sys
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
HW5_DIR = ROOT / "HW5" / "python"
OUTPUT_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(HW5_DIR))
import main as hw5_main

L = 128
q = 2
beta_c = np.log(1 + np.sqrt(2)) / 2
betas = np.arange(0.99, 0.85, -0.01) * beta_c
n_eq = 2000
n_samples = 300


if __name__ == "__main__":
    hw5_main.save_dir = str(HW5_DIR / "data")

    rows = []
    for beta in betas:
        state = hw5_main.equilibrate_wolff_state(L, q, beta, n_eq=n_eq, force=True)
        results = hw5_main.measure_wolff_observables(state, beta, q, n_samples=n_samples)
        rows.append(
            {
                "L": L,
                "q": q,
                "beta": float(beta),
                "specific_heat": float(results["specific_heat"]),
                "energy": float(results["energy"]),
                "mean_cluster_size": float(results["mean_cluster_size"]),
                "cluster_sizes": (results["cluster_sizes"]),
                "sigma_cluster_size": (results["sigma_cluster_size"])
            }
        )

    df = pd.DataFrame(rows)
    peak_idx = int(np.argmax(df["specific_heat"].to_numpy()))
    peak_row = df.iloc[peak_idx]

    peak_summary = pd.DataFrame(
        [
            {
                "L": L,
                "q": q,
                "beta_peak": float(peak_row["beta"]),
                "specific_heat_peak": float(peak_row["specific_heat"]),
                "energy_at_peak": float(peak_row["energy"]),
                "mean_cluster_size_at_peak": float(peak_row["mean_cluster_size"]),
            }
        ]
    )

    df.to_csv(OUTPUT_DIR / "hw8_problem16_single_L128_scan.csv", index=False)
    peak_summary.to_csv(OUTPUT_DIR / "hw8_problem16_single_L128_peak.csv", index=False)

    print("Saved scan data to", OUTPUT_DIR / "hw8_problem16_single_L128_scan.csv")
    print("Saved peak summary to", OUTPUT_DIR / "hw8_problem16_single_L128_peak.csv")
    print(df.to_string(index=False))
    print(peak_summary.to_string(index=False))
