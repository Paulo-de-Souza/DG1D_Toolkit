"""Cahn--Hilliard replica with the split LDG operator used in the toolkit."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dg1d_toolkit.dg1d_core import DGSpace1D
from dg1d_toolkit.dg1d_subcell import diffusion_rhs
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diffusion", choices=("ldg", "br2"), default="ldg")
    parser.add_argument("--quick", action="store_true", help="malha e tempo curtos")
    parser.add_argument("--output", type=Path, default=None)
    add_runtime_options(parser)
    args = parser.parse_args(argv)

    # Alberto's executed loop uses Nsteps=99000 and dt=2.2e-6, hence
    # t_final=0.2178; tb=[0,0.05] is not the actual loop duration.
    K, N, t_end, dt, source_steps = (
        (10, 1, 5.0e-4, 1.0e-4, 5)
        if args.quick else (40, 1, 99000 * 2.2e-6, 2.2e-6, 99000)
    )
    K, N, t_end, dt, steps = apply_runtime_overrides(
        args, K, N, t_end, dt, source_steps
    )
    gamma = (0.1**2) / 6.0
    mobility = 1.0
    space = DGSpace1D(K, N, 0.0, 1.0, quad_type="GL", P=2 * N, nlP=4 * N)
    # Deterministic analogue of Alberto's fluctuating IC; it avoids the
    # external icr_u.npy dependency while retaining the phase-field character.
    phi0 = 0.5 + 0.05 * np.cos(2.0 * math.pi * space.xc) + 0.005 * np.sin(6.0 * math.pi * space.xc)
    initial = project(space, phi0)
    unit = np.ones((space.nip, space.K + 2))
    faces = np.ones((2, space.K + 2))

    def laplace(field):
        if args.diffusion == "ldg":
            return diffusion_rhs(space, padded(space, field, periodic=True), unit, faces,
                                 method="ldg", periodic=True)
        return diffusion_rhs(space, padded(space, field, periodic=True), method="br2",
                             coefficient=1.0, eta=2.0, periodic=True)

    def rhs(states, time):
        del time
        phi = states[0]
        phi_h = np.dot(space.nlpsi, phi)
        free_energy = project(space, phi_h**3 - phi_h)
        chemical_potential = free_energy - gamma * laplace(phi)
        return [mobility * laplace(chemical_potential)]

    history = StateHistory(space, args.history_every) if args.history_every else None
    state, t, steps = integrate(
        [initial], rhs, t_end=t_end, dt=dt, method="rk4",
        diagnostics=history, progress_every=args.print_every,
        progress_name="Cahn-Hilliard", nsteps=steps,
    )
    if history is not None:
        history.finalize(state, t, steps)
    phi = np.dot(space.psi, state[0])
    mass0 = float(np.sum(space.J[None, :] * space.wi[:, None] * np.dot(space.psi, initial)))
    mass = float(np.sum(space.J[None, :] * space.wi[:, None] * phi))
    output = args.output or Path(__file__).resolve().parent / "results" / f"cahn_hilliard_{args.diffusion}.npz"
    save_result(output, space, state, time=t, mass_initial=mass0, mass_final=mass,
                diffusion=args.diffusion)
    figure = output.with_suffix(".png")
    if not args.no_plot:
        plot_profiles(figure, space, state, ["phi"], f"Cahn-Hilliard — {args.diffusion.upper()}",
                      initial_state=[initial], ylabels=["phi(x,t)"])
        if history is not None and len(history.times) > 1:
            plot_evolution(
                figure.with_name(figure.stem + "_evolution.png"),
                space.xc.T.reshape(-1), history.times,
                history.physical_arrays(), ["phi(x,t)"],
                f"Cahn-Hilliard — evolução temporal ({args.diffusion.upper()})",
            )
    print_summary(f"Cahn-Hilliard ({args.diffusion})", space, state, t, steps)
    print(f"  massa_inicial={mass0:.8e}; massa_final={mass:.8e}; resultado={output}; figura={figure if not args.no_plot else 'desativada'}")


if __name__ == "__main__":
    main()
