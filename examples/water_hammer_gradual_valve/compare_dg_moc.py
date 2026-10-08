"""Compare the saved DG and MOC results for the gradual-valve test.

Run the DG and MOC scripts first. If generated files are absent, the script
uses the versioned reference arrays distributed with the repository.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


EXAMPLE_DIR = Path(__file__).resolve().parent
GENERATED_DIR = EXAMPLE_DIR / "results" / "generated"
REFERENCE_DIR = EXAMPLE_DIR / "results" / "reference_data"
GENERATED_DIR.mkdir(parents=True, exist_ok=True)


def find_result(filename: str) -> Path:
    generated = GENERATED_DIR / filename
    return generated if generated.exists() else REFERENCE_DIR / filename


def plot_pair(t_ref, y_ref, t_dg, y_dg, title, ylabel, filename, sensor):
    fig, axis = plt.subplots(figsize=(10, 5), dpi=120)
    axis.plot(t_ref, y_ref, "b-", lw=2, label=f"MOC reference — {sensor}")
    axis.plot(t_dg, y_dg, "r--", lw=2, label=f"DG1D Toolkit — {sensor}")
    axis.set(title=title, xlabel="Time [s]", ylabel=ylabel, xlim=(0.0, 6.0))
    axis.grid(True)
    axis.legend(loc="lower left")
    fig.savefig(GENERATED_DIR / filename, dpi=180, bbox_inches="tight")


def main() -> None:
    dg = np.load(find_result("resultados_DG.npz"))
    moc = np.load(find_result("resultados_MOC.npz"))
    t_dg, t_moc = dg["tempo"], moc["tempo"]

    plot_pair(t_moc, moc["H_50"], t_dg, dg["H_50"], "Piezometric head: DG vs MOC", "Head [m]", "moc_vs_dg_H_sensor50.png", "x = 50 m")
    plot_pair(t_moc, moc["H_200"], t_dg, dg["H_200"], "Piezometric head: DG vs MOC", "Head [m]", "moc_vs_dg_H_sensor200.png", "x = 200 m")
    plot_pair(t_moc, moc["V_50"], t_dg, dg["V_50"], "Velocity: DG vs MOC", "Velocity [m/s]", "moc_vs_dg_V_sensor50.png", "x = 50 m")
    plot_pair(t_moc, moc["V_200"], t_dg, dg["V_200"], "Velocity: DG vs MOC", "Velocity [m/s]", "moc_vs_dg_V_sensor200.png", "x = 200 m")
    plt.show()


if __name__ == "__main__":
    main()
