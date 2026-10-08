"""Compare the instantaneous-valve DG trace with the lossless analytical trace."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


L = 800.0
a = 1000.0
g = 9.81
H_res = 20.0
V0 = 0.15
tmax = 15.0

EXAMPLE_DIR = Path(__file__).resolve().parent
GENERATED_DIR = EXAMPLE_DIR / "results" / "generated"
REFERENCE_DIR = EXAMPLE_DIR / "results" / "reference_data"
GENERATED_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    delta_head = a * V0 / g
    half_period = 2.0 * L / a
    full_period = 4.0 * L / a
    t_exact = np.linspace(0.0, tmax, 10_000)
    head_exact = np.where(
        t_exact == 0.0,
        H_res,
        np.where((t_exact % full_period) < half_period, H_res + delta_head, H_res - delta_head),
    )

    result_path = GENERATED_DIR / "resultados_DG_WHv2.npz"
    if not result_path.exists():
        result_path = REFERENCE_DIR / "resultados_DG_WHv2.npz"
    dg = np.load(result_path)

    fig, axis = plt.subplots(figsize=(10, 5), dpi=120)
    axis.plot(t_exact, head_exact, "k-", lw=1.5, label="Lossless analytical solution")
    axis.plot(dg["tempo"], dg["H"], "r--", lw=1.5, label="DG1D Toolkit")
    axis.set(
        title="Instantaneous valve closure: analytical trace vs DG",
        xlabel="Time [s]",
        ylabel="Piezometric head [m]",
        xlim=(0.0, tmax),
        ylim=(0.0, 50.0),
    )
    axis.grid(True)
    axis.legend()
    fig.savefig(GENERATED_DIR / "analytical_vs_dg_head.png", dpi=180, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
