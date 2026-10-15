"""Two-field Cahn--Hilliard test following Alberto's active DG script."""

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
)


def _padded_source_state(space, modal):
    """Use Alberto's one-sided Neumann ghost convention."""
    padded = np.zeros((space.Nldof, space.K + 2), dtype=float)
    padded[:, 1:-1] = modal
    right_trace = np.dot(space.Frk[0, :], modal[:, -1])
    padded[0, -1] = right_trace
    return padded


def _find_initial_file(name, explicit):
    """Use an explicit path or find Alberto's file in common locations."""
    if explicit is not None:
        return explicit
    candidates = (
        Path.cwd() / name,
        Path(__file__).resolve().parent / name,
    )
    return next((candidate for candidate in candidates if candidate.exists()), None)


def _load_or_build_initial(space, path, fallback):
    """Load one of Alberto's IC files or use a reproducible fallback."""
    if path is not None:
        raw = np.asarray(np.load(path), dtype=float)
        if raw.size != space.K * space.nip:
            raise ValueError(
                f"{path} deve conter {space.K * space.nip} valores "
                f"(K={space.K}, nip={space.nip})."
            )
        physical = raw.reshape(space.K, space.nip).T
    else:
        physical = fallback(space.xc)
    return project(space, physical)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--diffusion", choices=("ldg",), default="ldg",
        help="o script ativo do Alberto usa a formulação LDG acoplada",
    )
    parser.add_argument("--initial-u", type=Path, default=None,
                        help="arquivo icr_u.npy do caso original")
    parser.add_argument("--initial-v", type=Path, default=None,
                        help="arquivo icr_v.npy do caso original")
    parser.add_argument("--quick", action="store_true", help="malha e tempo curtos")
    parser.add_argument("--output", type=Path, default=None)
    add_runtime_options(parser)
    args = parser.parse_args(argv)

    # Effective production loop in Cahn-Hilliard-1D-Modal-DG.py.
    K, N, t_end, dt, source_steps = (
        (10, 1, 5.0e-4, 1.0e-4, 5)
        if args.quick else (40, 1, 99000 * 2.2e-6, 2.2e-6, 99000)
    )
    K, N, t_end, dt, steps = apply_runtime_overrides(
        args, K, N, t_end, dt, source_steps
    )

    # Parameters of Alberto's active fluctuating Cahn--Hilliard case.
    theta = 0.1
    alpha_fe = 3.0
    beta_fe = -1.0
    epsilon = 0.1
    gamma = (1.0 / 6.0) * epsilon**2
    rho = 0.08
    space = DGSpace1D(K, N, 0.0, 1.0, quad_type="GL", P=2 * N, nlP=4 * N)

    initial_u_path = _find_initial_file("icr_u.npy", args.initial_u)
    initial_v_path = _find_initial_file("icr_v.npy", args.initial_v)
    u_initial = _load_or_build_initial(
        space, initial_u_path,
        lambda x: 0.55 - np.exp(-3.0) * np.sin(6.0 * np.pi * x),
    )
    v_initial = _load_or_build_initial(
        space, initial_v_path,
        lambda x: np.zeros_like(x),
    )

    def rhs(states, time):
        del time
        u_modal, v_modal = states
        u_pad = _padded_source_state(space, u_modal)
        v_pad = _padded_source_state(space, v_modal)

        # Free-energy derivatives at the over-integration points.
        u_h = np.dot(space.nlpsi, u_modal)
        v_h = np.dot(space.nlpsi, v_modal)
        plus = (u_h + v_h) / (1.0 - u_h - v_h)
        minus = (u_h - v_h) / (1.0 - u_h + v_h)
        if np.any(plus <= 0.0) or np.any(minus <= 0.0):
            raise FloatingPointError(
                "o estado saiu do domínio logarítmico da energia livre; "
                "reduza dt ou verifique a condição inicial."
            )
        du_phi = theta * (np.log(plus) + np.log(minus))
        dv_phi = theta * (np.log(plus) - np.log(minus))
        du_psi = du_phi + 0.5 * alpha_fe * (1.0 - 2.0 * u_h)
        dv_psi = dv_phi - beta_fe * v_h
        r1 = project(space, du_psi)
        r2 = project(space, dv_psi)

        # Neumann condition for the chemical-potential component r1.
        r1_pad = np.zeros_like(u_pad)
        r2_pad = np.zeros_like(v_pad)
        r1_pad[:, 1:-1] = r1
        r2_pad[:, 1:-1] = r2
        r1_pad[0, -1] = np.dot(space.Frk[0, :], r1[:, -1])

        # First LDG gradients w1=du/dx and w2=dv/dx.
        w1 = np.zeros_like(u_pad)
        w2 = np.zeros_like(v_pad)
        w1[:, 1:-1] = space.InvM[:, None] * space.J[None, :] ** (-1) * (
            -np.dot(space.S.T, u_pad[:, 1:-1])
            + np.dot(space.Flkp1, u_pad[:, 2:])
            - np.dot(space.Flk, u_pad[:, 1:-1])
        )
        w2[:, 1:-1] = space.InvM[:, None] * space.J[None, :] ** (-1) * (
            -np.dot(space.S.T, v_pad[:, 1:-1])
            + np.dot(space.Flkp1, v_pad[:, 2:])
            - np.dot(space.Flk, v_pad[:, 1:-1])
        )

        # q1=gamma*d2u/dx2 and q2=gamma*d2v/dx2.
        q1 = np.zeros_like(u_pad)
        q2 = np.zeros_like(v_pad)
        q1[:, 1:-1] = space.InvM[:, None] * space.J[None, :] ** (-1) * gamma * (
            -np.dot(space.S.T, w1[:, 1:-1])
            + np.dot(space.Frk, w1[:, 1:-1])
            - np.dot(space.Frkm1, w1[:, :-2])
        )
        q2[:, 1:-1] = space.InvM[:, None] * space.J[None, :] ** (-1) * gamma * (
            -np.dot(space.S.T, w2[:, 1:-1])
            + np.dot(space.Frk, w2[:, 1:-1])
            - np.dot(space.Frkm1, w2[:, :-2])
        )

        q1_right = space.InvM * (space.J[-1] ** -1) * gamma * (
            -np.dot(space.S.T, w1[:, -1])
            + np.dot(space.Frk, w1[:, -1])
            - np.dot(space.Frkm1, w1[:, -2])
        )
        q1[0, -1] = np.dot(space.Frk[0, :], q1_right)

        # phat = d/dx(q1-r1), assembled with Alberto's LDG signs.
        phat = np.zeros_like(u_pad)
        phat[:, 1:-1] = space.InvM[:, None] * space.J[None, :] ** (-1) * (
            np.dot(space.S.T, q1[:, 1:-1])
            - np.dot(space.S.T, r1_pad[:, 1:-1])
            - np.dot(space.Flkp1, q1[:, 2:])
            + np.dot(space.Flk, q1[:, 1:-1])
            + np.dot(space.Flkp1, r1_pad[:, 2:])
            - np.dot(space.Flk, r1_pad[:, 1:-1])
        )

        # Variable mobility b(u,v)=u(1-u)(1/4-v^2).
        ph_h = np.dot(space.nlpsi, phat[:, 1:-1])
        r2_h = np.dot(space.nlpsi, r2_pad[:, 1:-1])
        q2_h = np.dot(space.nlpsi, q2[:, 1:-1])
        mobility = u_h * (1.0 - u_h) * (0.25 - v_h**2)
        s1 = project(space, mobility * ph_h)
        s2 = project(space, -mobility * (-q2_h + r2_h) / rho)

        # v_t = s2 and u_t = d/dx(s1).
        s1_pad = np.zeros_like(u_pad)
        s1_pad[:, 1:-1] = s1
        du_dt = space.InvM[:, None] * space.J[None, :] ** (-1) * (
            -np.dot(space.S.T, s1)
            + np.dot(space.Frk, s1)
            - np.dot(space.Frkm1, s1_pad[:, :-2])
        )
        dv_dt = s2
        return [du_dt, dv_dt]

    history = StateHistory(space, args.history_every) if args.history_every else None
    state, t, steps = integrate(
        [u_initial, v_initial], rhs, t_end=t_end, dt=dt, method="rk54",
        diagnostics=history, progress_every=args.print_every,
        progress_name="Cahn-Hilliard", nsteps=steps,
    )
    if history is not None:
        history.finalize(state, t, steps)

    u_final = np.dot(space.psi, state[0])
    u0 = np.dot(space.psi, u_initial)
    mass_u0 = float(np.sum(space.J[None, :] * space.wi[:, None] * u0))
    mass_uf = float(np.sum(space.J[None, :] * space.wi[:, None] * u_final))
    output = args.output or Path(__file__).resolve().parent / "results" / "cahn_hilliard_ldg.npz"
    save_result(
        output, space, state, time=t,
        mass_u_initial=mass_u0, mass_u_final=mass_uf,
        theta=theta, alpha_fe=alpha_fe, beta_fe=beta_fe,
        gamma=gamma, rho=rho,
        initial_u_file=str(initial_u_path or "fallback"),
        initial_v_file=str(initial_v_path or "fallback"),
    )
    figure = output.with_suffix(".png")
    if not args.no_plot:
        plot_profiles(
            figure, space, state, ["u", "v"], "Cahn-Hilliard — LDG",
            initial_state=[u_initial, v_initial], ylabels=["u(x,t)", "v(x,t)"],
        )
        if history is not None and len(history.times) > 1:
            plot_evolution(
                figure.with_name(figure.stem + "_evolution.png"),
                space.xc.T.reshape(-1), history.times,
                history.physical_arrays(), ["u(x,t)", "v(x,t)"],
                "Cahn-Hilliard — evolução temporal (LDG)",
            )
    print_summary("Cahn-Hilliard (LDG)", space, state, t, steps)
    print(
        f"  formulação=dois campos + energia livre logarítmica + mobilidade variável; "
        f"massa_u=[{mass_u0:.8e},{mass_uf:.8e}]; resultado={output}; "
        f"figura={figure if not args.no_plot else 'desativada'}"
    )


if __name__ == "__main__":
    main()
