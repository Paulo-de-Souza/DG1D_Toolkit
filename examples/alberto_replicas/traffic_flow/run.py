"""Traffic-flow Green-Light test with the source script's AV parameters."""

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
    integrate,
    plot_evolution,
    plot_profiles,
    print_summary,
    project,
    save_result,
    scalar_rhs,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    add_runtime_options(parser)
    args = parser.parse_args(argv)

    # These are the effective values in Alberto's executed loop:
    # Nsteps=4000 and dt=0.00025, therefore t_final=1.0.
    K, N, t_end, dt, source_steps = (
        (12, 3, 0.01, 5.0e-4, 20)
        if args.quick else (39, 9, 1.0, 2.5e-4, 4000)
    )
    K, N, t_end, dt, steps = apply_runtime_overrides(
        args, K, N, t_end, dt, source_steps
    )
    Vm, um = 1.0, 10.0
    space = DGSpace1D(K, N, 0.0, 4.0, quad_type="GL", P=2 * N, nlP=3 * N)
    initial_physical = np.zeros_like(space.xc)
    initial_physical[:, : K // 2] = (um / 2.0) * space.xc[:, : K // 2]
    initial = project(space, initial_physical)

    def flux(value):
        return Vm * (value - value**2 / um)

    def speed(value):
        return Vm * (1.0 - 2.0 * value / um)

    def rhs(states, time):
        del time
        result, _ = scalar_rhs(
            space, states[0], flux, speed,
            boundary_values=(0.0, 0.0), artificial=True,
            sensor_constant=1.0, viscosity_factor=0.5,
            profile="linear", diffusion="ldg",
        )
        return [result]

    history = StateHistory(space, args.history_every) if args.history_every else None
    state, t, steps = integrate(
        [initial], rhs, t_end=t_end, dt=dt, method="rk54",
        diagnostics=history, progress_every=args.print_every,
        progress_name="Traffic-Flow", nsteps=steps,
    )
    if history is not None:
        history.finalize(state, t, steps)
    output = args.output or Path(__file__).resolve().parent / "results" / "traffic_flow.npz"
    save_result(output, space, state, time=t, Vm=Vm, um=um, sensor_constant=1.0)
    figure = output.with_suffix(".png")
    if not args.no_plot:
        plot_profiles(figure, space, state, ["rho"], "Traffic Flow — Green Light",
                      initial_state=[initial], ylabels=["density u(x,t)"])
        if history is not None and len(history.times) > 1:
            plot_evolution(
                figure.with_name(figure.stem + "_evolution.png"),
                space.xc.T.reshape(-1), history.times,
                history.physical_arrays(), ["density u(x,t)"],
                "Traffic Flow — evolução temporal",
            )
    print_summary("Traffic-Flow", space, state, t, steps)
    print(f"  captura=Persson/Warburton + LDG; BC Green-Light; resultado={output}; figura={figure if not args.no_plot else 'desativada'}")


if __name__ == "__main__":
    main()
