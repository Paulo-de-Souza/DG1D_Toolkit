"""Reflective Shu--Osher Euler test with sub-cell-style artificial viscosity."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dg1d_toolkit.dg1d_core import DGSpace1D
from examples.alberto_replicas.common import (
    StateHistory,
    add_runtime_options,
    apply_runtime_overrides,
    euler_rhs,
    integrate,
    plot_arrays,
    plot_evolution,
    print_summary,
    project,
    save_result,
)


def conserved(rho, velocity, pressure, gamma):
    momentum = rho * velocity
    energy = pressure / (gamma - 1.0) + 0.5 * momentum**2 / rho
    return rho, momentum, energy


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--reference", action="store_true", help="configuração K=120,N=5,t=0.3,dt=3e-6")
    parser.add_argument("--output", type=Path, default=None)
    add_runtime_options(parser)
    args = parser.parse_args(argv)

    # The active Alberto script advances exactly 100000 times with dt=3e-6,
    # so its effective final time is 0.3.
    if args.reference or not args.quick:
        K, N, t_end, dt, source_steps = 120, 5, 0.3, 3.0e-6, 100000
    else:
        K, N, t_end, dt, source_steps = 24, 2, 0.003, 2.0e-5, 150
    K, N, t_end, dt, steps = apply_runtime_overrides(
        args, K, N, t_end, dt, source_steps
    )
    gamma = 1.4
    space = DGSpace1D(K, N, 0.0, 1.0, quad_type="GL", P=2 * N, nlP=3 * N)
    x = space.xc
    rho = np.where(x < 0.125, 3.857143, 1.0 + 0.2 * np.sin(16.0 * np.pi * x))
    velocity = np.where(x < 0.125, 2.629369, 0.0)
    pressure = np.where(x < 0.125, 10.333333, 1.0)
    states0 = [project(space, value) for value in conserved(rho, velocity, pressure, gamma)]
    left = conserved(3.857143, 2.629369, 10.333333, gamma)
    # The active Alberto reflective script uses a partially reflected right
    # state, rather than a periodic outlet.
    right = conserved(3.857143 / 2.0, -2.629369 / 2.0, 10.333333 / 2.0, gamma)

    def rhs(states, time):
        del time
        result, _ = euler_rhs(space, states, left, right, gamma=gamma,
                              artificial=True, sensor_constant=100.0,
                              profile="linear", diffusion="ldg")
        return result

    history = StateHistory(space, args.history_every) if args.history_every else None
    state, t, steps = integrate(
        states0, rhs, t_end=t_end, dt=dt, method="rk54",
        diagnostics=history, progress_every=args.print_every,
        progress_name="Shu-Osher", nsteps=steps,
    )
    if history is not None:
        history.finalize(state, t, steps)
    output = args.output or Path(__file__).resolve().parent / "results" / "shu_osher.npz"
    save_result(output, space, state, time=t, gamma=gamma, sensor_constant=100.0)
    figure = output.with_suffix(".png")
    if not args.no_plot:
        final_values = [np.dot(space.psi, modal).T.reshape(-1) for modal in state]
        rho_f, rhou_f, energy_f = final_values
        velocity_f = rhou_f / np.maximum(rho_f, 1.0e-10)
        pressure_f = (gamma - 1.0) * (energy_f - 0.5 * rhou_f**2 / np.maximum(rho_f, 1.0e-10))
        plot_arrays(figure, space.xc.T.reshape(-1), [rho_f, velocity_f, pressure_f],
                    ["rho", "u", "p"], "Shu-Osher — artificial viscosity",
                    ylabels=["rho", "u", "p"])
        if history is not None and len(history.times) > 1:
            rho_h, rhou_h, energy_h = history.physical_arrays()
            rho_safe = np.maximum(rho_h, 1.0e-10)
            velocity_h = rhou_h / rho_safe
            pressure_h = (gamma - 1.0) * (energy_h - 0.5 * rhou_h**2 / rho_safe)
            plot_evolution(
                figure.with_name(figure.stem + "_evolution.png"),
                space.xc.T.reshape(-1), history.times,
                [rho_h, velocity_h, pressure_h], ["rho", "u", "p"],
                "Shu-Osher — evolução temporal",
            )
    print_summary("Shu-Osher", space, state, t, steps)
    print(f"  captura=Persson/Warburton + LDG; BC direita=reflexiva parcial; resultado={output}; figura={figure if not args.no_plot else 'desativada'}")


if __name__ == "__main__":
    main()
