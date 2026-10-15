"""Shock-density Euler test with modal artificial-viscosity capture."""

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
    parser.add_argument("--reference", action="store_true", help="configuração K=80,N=5,t=1.8,dt=5.5e-5")
    parser.add_argument("--output", type=Path, default=None)
    add_runtime_options(parser)
    args = parser.parse_args(argv)

    # The active Alberto script advances exactly 32728 times with dt=5.5e-5.
    # Its effective final time is therefore 1.80004 (the nominal tb is 1.8).
    if args.reference or not args.quick:
        K, N, t_end, dt, source_steps = 80, 5, 32728 * 5.5e-5, 5.5e-5, 32728
    else:
        K, N, t_end, dt, source_steps = 16, 2, 0.004, 5.0e-4, 8
    K, N, t_end, dt, steps = apply_runtime_overrides(
        args, K, N, t_end, dt, source_steps
    )
    gamma = 1.4
    space = DGSpace1D(K, N, -5.0, 5.0, quad_type="GL", P=2 * N, nlP=3 * N)
    x = space.xc
    rho = np.where(x < -4.0, 3.857143, 1.0 + 0.2 * np.sin(5.0 * x))
    velocity = np.where(x < -4.0, 2.629369, 0.0)
    pressure = np.where(x < -4.0, 10.333333, 1.0)
    states0 = [project(space, value) for value in conserved(rho, velocity, pressure, gamma)]
    right = conserved(1.0 + 0.2 * np.sin(5.0 * space.xmax), 0.0, 1.0, gamma)
    left = conserved(3.857143, 2.629369, 10.333333, gamma)

    def rhs(states, time):
        del time
        result, _ = euler_rhs(space, states, left, right, gamma=gamma,
                              artificial=True, sensor_constant=5.0,
                              profile="linear", diffusion="ldg")
        return result

    history = StateHistory(space, args.history_every) if args.history_every else None
    state, t, steps = integrate(
        states0, rhs, t_end=t_end, dt=dt, method="rk54",
        diagnostics=history, progress_every=args.print_every,
        progress_name="Shock-Density", nsteps=steps,
    )
    if history is not None:
        history.finalize(state, t, steps)
    output = args.output or Path(__file__).resolve().parent / "results" / "shock_density.npz"
    save_result(output, space, state, time=t, gamma=gamma, sensor_constant=5.0)
    figure = output.with_suffix(".png")
    if not args.no_plot:
        final_values = [np.dot(space.psi, modal).T.reshape(-1) for modal in state]
        rho_f, rhou_f, energy_f = final_values
        pressure_f = (gamma - 1.0) * (energy_f - 0.5 * rhou_f**2 / np.maximum(rho_f, 1.0e-10))
        plot_arrays(figure, space.xc.T.reshape(-1), [rho_f, rhou_f / np.maximum(rho_f, 1.0e-10), pressure_f],
                    ["rho", "u", "p"], "Shock-Density — artificial viscosity",
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
                "Shock-Density — evolução temporal",
            )
    print_summary("Shock-Density", space, state, t, steps)
    print(f"  captura=Persson/Warburton + LDG; resultado={output}; figura={figure if not args.no_plot else 'desativada'}")


if __name__ == "__main__":
    main()
