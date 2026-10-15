"""Shallow-water Gaussian pulse with Alberto's generalized slope limiter."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dg1d_toolkit.dg1d_core import DGSpace1D
from dg1d_toolkit.dg1d_integrators import RKSSP54_Step
from dg1d_toolkit.dg1d_limiters import SlopeLimiterN
from dg1d_toolkit.numericalfluxes import rusanov
from dg1d_toolkit.operators import FluxProjection
from examples.alberto_replicas.common import (
    StateHistory,
    add_runtime_options,
    apply_runtime_overrides,
    dirichlet_padded,
    integrate,
    plot_evolution,
    plot_profiles,
    print_summary,
    project,
    save_result,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    add_runtime_options(parser)
    args = parser.parse_args(argv)

    K, N, t_end, dt, source_steps = (
        (8, 3, 0.02, 5.0e-4, 40)
        if args.quick else (20, 8, 1.0, 2.0e-4, 5000)
    )
    K, N, t_end, dt, steps = apply_runtime_overrides(
        args, K, N, t_end, dt, source_steps
    )
    gravity, H0, amplitude, sigma, center = 9.81, 0.005, 0.05, 0.05, 0.5
    # SlopeLimiterN in the current toolkit assumes nip=N+1, as in its GL
    # examples.  Keep the source PDE/IC/limiter and use GL until that limiter
    # is generalized to the N+2 GLL nodal layout.
    space = DGSpace1D(K, N, 0.0, 1.0, quad_type="GL", P=2 * N, nlP=4 * N)
    h0 = H0 * (1.0 + amplitude * np.exp(-((space.xc - center) ** 2) / sigma**2))
    state0 = [project(space, h0), project(space, np.zeros_like(h0))]

    def rhs(states, time):
        del time
        h, velocity = states
        hp = dirichlet_padded(space, h, H0, H0)
        up = dirichlet_padded(space, velocity, 0.0, 0.0)
        flux_h, flux_u = FluxProjection(
            space, [hp, up],
            lambda hh, uu: [hh * uu, gravity * hh + 0.5 * uu**2],
        )
        # Boundary flux means use the prescribed traces, as in the source.
        hL, hR = H0, H0
        uL, uR = 0.0, 0.0
        flux_h[0, 0] = hL * uL
        flux_h[0, -1] = hR * uR
        flux_u[0, 0] = gravity * hL + 0.5 * uL**2
        flux_u[0, -1] = gravity * hR + 0.5 * uR**2
        hnodal = np.dot(space.psi, h)
        unodal = np.dot(space.psi, velocity)
        speed = np.abs(unodal) + np.sqrt(gravity * np.maximum(hnodal, 1.0e-10))
        face_speed = np.empty(space.K + 1)
        face_speed[0], face_speed[-1] = speed[0, 0], speed[-1, -1]
        face_speed[1:-1] = np.maximum(speed[-1, :-1], speed[0, 1:])
        volume_h = np.dot(space.S.T, flux_h[:, 1:-1])
        volume_u = np.dot(space.S.T, flux_u[:, 1:-1])
        num_h = rusanov(space, hp, flux_h, face_speed)
        num_u = rusanov(space, up, flux_u, face_speed)
        return [
            space.InvM[:, None] * space.J[None, :] ** -1 * (volume_h + num_h),
            space.InvM[:, None] * space.J[None, :] ** -1 * (volume_u + num_u),
        ]

    limiter = SlopeLimiterN(space)
    history = StateHistory(space, args.history_every) if args.history_every else None
    state, t, steps = integrate(
        state0, rhs, t_end=t_end, dt=dt, method="rk54", limiter=limiter,
        diagnostics=history, progress_every=args.print_every,
        progress_name="Shallow-Water", nsteps=steps,
    )
    if history is not None:
        history.finalize(state, t, steps)
    output = args.output or Path(__file__).resolve().parent / "results" / "shallow_water.npz"
    save_result(output, space, state, time=t, gravity=gravity)
    figure = output.with_suffix(".png")
    if not args.no_plot:
        plot_profiles(figure, space, state, ["h", "u"], "Shallow-Water — SlopeLimiterN",
                      initial_state=state0, ylabels=["h(x,t)", "u(x,t)"])
        if history is not None and len(history.times) > 1:
            plot_evolution(
                figure.with_name(figure.stem + "_evolution.png"),
                space.xc.T.reshape(-1), history.times,
                history.physical_arrays(), ["h(x,t)", "u(x,t)"],
                "Shallow-Water — evolução temporal",
            )
    print_summary("Shallow-Water", space, state, t, steps)
    print(f"  limitador=SlopeLimiterN; resultado={output}; figura={figure if not args.no_plot else 'desativada'}")


if __name__ == "__main__":
    main()
