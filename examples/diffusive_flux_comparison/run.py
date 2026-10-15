"""Compare BR1 (old), LDG (new) and BR2 (new) on periodic heat diffusion.

Run from the toolkit repository root with::

    python -m examples.diffusive_flux_comparison.run --K 32 --t-end 0.05

The test uses the same modal DG space and RK4 integrator as the toolkit
examples.  The exact solution for ``u_t = nu*u_xx`` and
``u(x,0)=sin(2*pi*x)`` is used to report the error.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dg1d_toolkit.dg1d_core import DGSpace1D
from dg1d_toolkit.dg1d_integrators import RK4_Step
from dg1d_toolkit.dg1d_subcell import diffusion_rhs


def project(space, values):
    rhs = np.dot(space.wi * space.psi.T, values)
    return space.InvM[:, None] * rhs


def run_method(space, initial, *, method, coefficient, t_end, dt, eta):
    epst = np.full((space.nip, space.K + 2), coefficient, dtype=float)
    epsb = np.full((2, space.K + 2), coefficient, dtype=float)
    state = initial.copy()
    nsteps = int(np.ceil(t_end / dt))
    dt_used = t_end / nsteps
    t = 0.0
    for _ in range(nsteps):
        rhs = lambda states, time: [
            diffusion_rhs(
                space,
                states[0],
                epst,
                epsb,
                method=method,
                coefficient=coefficient,
                eta=eta,
                periodic=True,
            )
        ]
        state = RK4_Step([state], t, dt_used, rhs)[0]
        t += dt_used
    return state, nsteps


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--K", type=int, default=32)
    parser.add_argument("--N", type=int, default=3)
    parser.add_argument("--nu", type=float, default=2.0e-3)
    parser.add_argument("--t-end", type=float, default=0.05)
    parser.add_argument("--dt", type=float, default=1.0e-4)
    parser.add_argument("--eta", type=float, default=2.0)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args(argv)

    space = DGSpace1D(K=args.K, N=args.N, xmin=0.0, xmax=1.0)
    initial = project(space, np.sin(2.0 * np.pi * space.xc))
    methods = ("br1", "ldg", "br2")
    results = {}
    errors = {}

    for method in methods:
        state, nsteps = run_method(
            space,
            initial,
            method=method,
            coefficient=args.nu,
            t_end=args.t_end,
            dt=args.dt,
            eta=args.eta,
        )
        values = np.dot(space.psi, state).T.flatten()
        results[method] = values

    x = space.xc.T.flatten()
    exact = np.exp(-args.nu * (2.0 * np.pi) ** 2 * args.t_end) * np.sin(2.0 * np.pi * x)
    for method in methods:
        errors[method] = float(np.max(np.abs(results[method] - exact)))

    output_dir = Path(__file__).resolve().parent / "results" / "generated"
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(output_dir / "diffusive_flux_comparison.npz", x=x, exact=exact, **results)

    if not args.no_plot:
        plt.figure(figsize=(8, 4.5))
        plt.plot(x, exact, "k--", lw=1.8, label="exact")
        for method in methods:
            plt.plot(x, results[method], lw=1.2, label=method.upper())
        plt.xlabel("x")
        plt.ylabel("u")
        plt.title("Periodic diffusion: BR1, LDG and BR2")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "diffusive_flux_comparison.png", dpi=180)
        plt.close()

    print(f"completed {nsteps} RK4 steps")
    for method in methods:
        print(f"{method.upper()} max error: {errors[method]:.6e}")
    print(f"results: {output_dir}")


if __name__ == "__main__":
    main()
