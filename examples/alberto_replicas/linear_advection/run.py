"""Linear-advection step with the compact artificial-viscosity profile."""

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

    # Alberto's executed loop uses Nsteps=410 and dt=0.002, i.e. t=0.82.
    K, N, t_end, dt, source_steps = (
        (6, 4, 0.02, 1.0e-3, 20)
        if args.quick else (12, 10, 0.82, 0.002, 410)
    )
    K, N, t_end, dt, steps = apply_runtime_overrides(
        args, K, N, t_end, dt, source_steps
    )
    speed = 0.5
    space = DGSpace1D(K, N, 0.0, 4.0, quad_type="GL", P=2 * N, nlP=3 * N)
    initial = project(space, np.where((space.xc >= 1.0) & (space.xc <= 3.0), 1.0, 0.0))

    def rhs(states, time):
        del time
        result, _ = scalar_rhs(
            space, states[0], lambda value: speed * value,
            lambda value: speed + 0.0 * value,
            boundary_values=(0.0, 0.0), artificial=True,
            sensor_constant=0.01, viscosity_factor=0.5,
            profile="compact", diffusion="br1",
        )
        return [result]

    history = StateHistory(space, args.history_every) if args.history_every else None
    state, t, steps = integrate(
        [initial], rhs, t_end=t_end, dt=dt, method="rk54",
        diagnostics=history, progress_every=args.print_every,
        progress_name="Linear-Advection", nsteps=steps,
    )
    if history is not None:
        history.finalize(state, t, steps)
    output = args.output or Path(__file__).resolve().parent / "results" / "linear_advection.npz"
    save_result(output, space, state, time=t, speed=speed, sensor_constant=0.01)
    figure = output.with_suffix(".png")
    if not args.no_plot:
        x = space.xc.T.reshape(-1)
        exact = np.where((x - speed * t >= 1.0) & (x - speed * t <= 3.0), 1.0, 0.0)
        plot_profiles(figure, space, state, ["u"], "Linear Advection — compact capture",
                      references=[exact], ylabels=["u(x,t)"])
        if history is not None and len(history.times) > 1:
            plot_evolution(
                figure.with_name(figure.stem + "_evolution.png"),
                space.xc.T.reshape(-1), history.times,
                history.physical_arrays(), ["u(x,t)"],
                "Linear Advection — evolução temporal",
            )
    print_summary("Linear-Advection", space, state, t, steps)
    print(f"  captura=Persson/Warburton + perfil compacto; resultado={output}; figura={figure if not args.no_plot else 'desativada'}")


if __name__ == "__main__":
    main()
