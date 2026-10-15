"""Allen--Cahn replica: smooth LDG/BR2 diffusion test from Alberto's scripts."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dg1d_toolkit.dg1d_subcell import diffusion_rhs
from dg1d_toolkit.operators import FluxProjection
from examples.alberto_replicas.common import (
    StateHistory,
    add_runtime_options,
    apply_runtime_overrides,
    integrate,
    padded,
    plot_evolution,
    plot_profiles,
    print_summary,
    project,
    save_result,
)
from dg1d_toolkit.dg1d_core import DGSpace1D


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diffusion", choices=("ldg", "br2"), default="ldg")
    parser.add_argument("--quick", action="store_true", help="malha curta para teste")
    parser.add_argument("--output", type=Path, default=None)
    add_runtime_options(parser)
    args = parser.parse_args(argv)

    # Alberto's executed loop uses Nsteps=20000 and dt=0.00025, so its
    # effective final time is 5.0 despite tb=[0,0.05] in the setup block.
    K, N, t_end, dt, source_steps = (
        (10, 2, 0.005, 5.0e-4, 10)
        if args.quick else (20, 3, 5.0, 2.5e-4, 20000)
    )
    K, N, t_end, dt, steps = apply_runtime_overrides(
        args, K, N, t_end, dt, source_steps
    )
    mu, beta, eps = 1.0, 0.5, math.sqrt(6.0)
    space = DGSpace1D(K, N, 0.0, 4.0 * math.pi, quad_type="GL", P=2 * N, nlP=4 * N)
    initial = project(space, np.cos(0.5 * space.xc))

    def rhs(states, time):
        del time
        u = states[0]
        u_pad = padded(space, u, periodic=True)
        # Alberto's active smooth formulation uses f_h=-beta*u, hence
        # -mu*f_h is the reaction +mu*beta*u.
        FluxProjection(space, [u_pad], lambda value: -beta * value)
        if args.diffusion == "ldg":
            unit = np.ones((space.nip, space.K + 2))
            faces = np.ones((2, space.K + 2))
            laplace = diffusion_rhs(
                space, u_pad, unit * (mu * eps**2), faces * (mu * eps**2),
                method="ldg", periodic=True,
            )
        else:
            laplace = diffusion_rhs(
                space, u_pad, method="br2", coefficient=mu * eps**2,
                eta=2.0, periodic=True,
            )
        return [mu * beta * u + laplace]

    history = StateHistory(space, args.history_every) if args.history_every else None
    state, t, steps = integrate(
        [initial], rhs, t_end=t_end, dt=dt, method="rk4",
        diagnostics=history, progress_every=args.print_every,
        progress_name="Allen-Cahn", nsteps=steps,
    )
    if history is not None:
        history.finalize(state, t, steps)
    x = space.xc.T.reshape(-1)
    exact = np.exp(-t) * np.cos(0.5 * x)
    numerical = np.dot(space.psi, state[0]).T.reshape(-1)
    error = float(np.max(np.abs(numerical - exact)))
    output = args.output or Path(__file__).resolve().parent / "results" / f"allen_cahn_{args.diffusion}.npz"
    save_result(output, space, state, time=t, error_inf=error, diffusion=args.diffusion)
    figure = output.with_suffix(".png")
    if not args.no_plot:
        plot_profiles(figure, space, state, ["u"], f"Allen-Cahn — {args.diffusion.upper()}",
                      references=[exact], ylabels=["u(x,t)"])
        if history is not None and len(history.times) > 1:
            plot_evolution(
                figure.with_name(figure.stem + "_evolution.png"),
                space.xc.T.reshape(-1), history.times,
                history.physical_arrays(), ["u(x,t)"],
                f"Allen-Cahn — evolução temporal ({args.diffusion.upper()})",
            )
    print_summary(f"Allen-Cahn ({args.diffusion})", space, state, t, steps)
    print(f"  erro_infinito={error:.6e}; resultado={output}; figura={figure if not args.no_plot else 'desativada'}")


if __name__ == "__main__":
    main()
