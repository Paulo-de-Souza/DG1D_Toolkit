"""Compare the original hierarchical limiter with artificial-viscosity DG.

Run from the toolkit repository root with::

    python -m examples.shock_capture_burgers.run --K 80 --t-end 0.25

The artificial-viscosity branch uses the new Persson/Warburton sensor, the
smoothed ``epsilon`` field, the modified-stiffness LDG operator and the old
Lax--Friedrichs flux/projection machinery.  The limiter branch is the
pre-existing toolkit path.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dg1d_toolkit.dg1d_capture_shock import artificial_viscosity
from dg1d_toolkit.dg1d_core import DGSpace1D
from dg1d_toolkit.dg1d_integrators import RKSSP104_Step
from dg1d_toolkit.dg1d_limiters import HierarchicalLimiter
from dg1d_toolkit.dg1d_subcell import diffusion_rhs, pad_periodic
from dg1d_toolkit.numericalfluxes import lax_friedrichs
from dg1d_toolkit.operators import FluxProjection


def f_burgers(u):
    return 0.5 * u**2


def project(space, values):
    rhs = np.dot(space.wi * space.psi.T, values)
    return space.InvM[:, None] * rhs


def rhs_burgers(U_modal, t, space, *, mode="artificial", diagnostics=None):
    del t
    padded = pad_periodic(space, U_modal)
    ft = FluxProjection(space, [padded], f_burgers)[0]
    ft[:, 0] = f_burgers(padded[0, 0])
    ft[:, -1] = f_burgers(padded[0, -1])

    values = np.dot(space.psi, U_modal)
    speed = float(np.max(np.abs(values)))
    advective = space.InvM[:, None] * space.J[None, :] ** (-1) * (
        np.dot(space.S.T, ft[:, 1:-1]) + lax_friedrichs(space, padded, ft, speed)
    )

    if mode == "limiter":
        if diagnostics is not None:
            diagnostics["viscosity"] = np.zeros(space.K)
            diagnostics["activation"] = np.zeros(space.K, dtype=bool)
        return advective

    sensor, epst, epsb = artificial_viscosity(
        space,
        U_modal,
        wave_speed=np.max(np.abs(values), axis=0),
        sensor="persson",
        periodic=True,
    )
    if diagnostics is not None and sensor is not None:
        diagnostics["viscosity"] = sensor.viscosity.copy()
        diagnostics["activation"] = sensor.flagged.copy()

    # This is the active Alberto branch: variable epsilon + modified
    # stiffness matrices + Hesthaven--Warburton LDG interface terms.
    dissipative = diffusion_rhs(
        space,
        U_modal,
        epst,
        epsb,
        method="ldg",
        periodic=True,
    )
    return advective + dissipative


def run_case(space, initial, *, mode, t_end, dt):
    state = initial.copy()
    diagnostics = {}
    limiter = HierarchicalLimiter(space) if mode == "limiter" else None
    nsteps = int(np.ceil(t_end / dt))
    dt_used = t_end / nsteps
    t = 0.0
    for _ in range(nsteps):
        rhs = lambda states, time: [
            rhs_burgers(states[0], time, space, mode=mode, diagnostics=diagnostics)
        ]
        state = RKSSP104_Step([state], t, dt_used, rhs, limiter)[0]
        t += dt_used
    diagnostics["nsteps"] = nsteps
    return state, diagnostics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--K", type=int, default=80)
    parser.add_argument("--N", type=int, default=3)
    parser.add_argument("--t-end", type=float, default=0.25)
    parser.add_argument("--dt", type=float, default=1.0e-3)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args(argv)

    space = DGSpace1D(K=args.K, N=args.N, xmin=0.0, xmax=1.0)
    initial = project(space, 0.5 + np.sin(2.0 * np.pi * space.xc))

    limiter_state, limiter_info = run_case(
        space, initial, mode="limiter", t_end=args.t_end, dt=args.dt
    )
    av_state, av_info = run_case(
        space, initial, mode="artificial", t_end=args.t_end, dt=args.dt
    )

    x = space.xc.T.flatten()
    limiter_values = np.dot(space.psi, limiter_state).T.flatten()
    av_values = np.dot(space.psi, av_state).T.flatten()

    output_dir = Path(__file__).resolve().parent / "results" / "generated"
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_dir / "burgers_shock_capture.npz",
        x=x,
        limiter=limiter_values,
        artificial_viscosity=av_values,
        viscosity=av_info.get("viscosity", np.zeros(space.K)),
        activated=av_info.get("activation", np.zeros(space.K, dtype=bool)),
    )

    if not args.no_plot:
        plt.figure(figsize=(8, 4.5))
        plt.plot(x, limiter_values, label="hierarchical limiter", lw=1.6)
        plt.plot(x, av_values, label="artificial viscosity + LDG", lw=1.6)
        plt.xlabel("x")
        plt.ylabel("u")
        plt.title("Inviscid Burgers: limiter versus shock-capturing viscosity")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "burgers_shock_capture.png", dpi=180)
        plt.close()

    nflag = int(np.count_nonzero(av_info.get("activation", [])))
    print(f"completed {limiter_info['nsteps']} steps")
    print(f"artificial-viscosity cells active at final stage: {nflag}/{space.K}")
    print(f"results: {output_dir}")


if __name__ == "__main__":
    main()
