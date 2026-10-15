"""Standalone Allen--Cahn reproduction of Alberto's smooth test.

The numerical parameters and the initial condition are the effective values
used by the corresponding script in ``examples/alberto_replicas``.  This file
keeps the complete assembly visible: modal projection, over-integrated flux
projection, DG diffusion, RK4 time integration, diagnostic history, PNG plots
and GIF generation.

From the repository root, run for example::

    python examples/allen_cahn/run.py --quick
    python examples/allen_cahn/run.py --diffusion ldg --print-every 1000

The full case has 20,000 steps, as in Alberto's executed loop.  ``--quick``
is only a smoke test and does not change the production preset.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dg1d_toolkit.dg1d_core import DGSpace1D
from dg1d_toolkit.dg1d_integrators import RK4_Step
from dg1d_toolkit.dg1d_postprocessing import create_gif
from dg1d_toolkit.dg1d_subcell import diffusion_rhs, pad_periodic
from dg1d_toolkit.operators import FluxProjection


def project(space: DGSpace1D, values: np.ndarray) -> np.ndarray:
    """L2-project values sampled at the standard or nonlinear quadrature."""
    values = np.asarray(values, dtype=float)
    if values.shape == (space.nip, space.K):
        basis, weights = space.psi, space.wi
    elif values.shape == (space.nlnip, space.K):
        basis, weights = space.nlpsi, space.nlwi
    else:
        raise ValueError(
            f"values deve ter shape {(space.nip, space.K)} ou "
            f"{(space.nlnip, space.K)}; recebido {values.shape}."
        )
    rhs = np.dot(weights[None, :] * basis.T, values)
    return space.InvM[:, None] * rhs


class History:
    """Store physical snapshots without coupling the example to ``common``."""

    def __init__(self, space: DGSpace1D, every: int):
        self.space = space
        self.every = max(int(every), 1)
        self.times: list[float] = []
        self.values: list[np.ndarray] = []

    def record(self, state: list[np.ndarray], time: float, step: int) -> None:
        if step == 0 or step % self.every == 0:
            self.times.append(float(time))
            self.values.append(self.physical(state[0]))

    def finalize(self, state: list[np.ndarray], time: float) -> None:
        if not self.times or abs(self.times[-1] - time) > 1.0e-14:
            self.times.append(float(time))
            self.values.append(self.physical(state[0]))

    def physical(self, modal: np.ndarray) -> np.ndarray:
        return np.dot(self.space.psi, modal).T.reshape(-1)

    def array(self) -> np.ndarray:
        return np.stack(self.values, axis=0)


def integrate(
    initial: list[np.ndarray],
    rhs,
    *,
    t_end: float,
    dt: float,
    steps: int,
    history: History,
    print_every: int,
) -> tuple[list[np.ndarray], float, int]:
    """Advance with the toolkit RK4 stepper and preserve Alberto's loop."""
    state = [np.array(value, dtype=float, copy=True) for value in initial]
    time = 0.0
    actual_steps = 0
    history.record(state, time, 0)
    for step in range(1, steps + 1):
        h = min(dt, max(t_end - time, 0.0))
        if h == 0.0:
            break
        state = RK4_Step(state, time, h, rhs)
        time += h
        actual_steps = step
        history.record(state, time, step)
        if print_every and (step == 1 or step % print_every == 0 or step == steps):
            print(f"[Allen-Cahn] step {step:>7d}/{steps:<7d}  t={time:.8g}/{t_end:.8g}")
    history.finalize(state, time)
    return state, time, actual_steps


def plot_final(path: Path, x: np.ndarray, initial: np.ndarray, final: np.ndarray,
               exact: np.ndarray, diffusion: str, time: float) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.plot(x, initial, "--", color="0.45", lw=1.2, label="inicial")
    ax.plot(x, final, color="tab:blue", lw=1.6, label="DG final")
    ax.plot(x, exact, ":", color="tab:red", lw=1.4, label="solução exata")
    ax.set_xlabel("x")
    ax.set_ylabel("u(x,t)")
    ax.set_title(f"Allen–Cahn — {diffusion.upper()} — t={time:g}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_evolution(path: Path, x: np.ndarray, times: np.ndarray,
                   values: np.ndarray, diffusion: str) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    image = ax.imshow(
        values,
        origin="lower",
        aspect="auto",
        extent=(float(x.min()), float(x.max()), float(times[0]), float(times[-1])),
        cmap="viridis",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_title(f"Allen–Cahn — mapa espaço–tempo ({diffusion.upper()})")
    fig.colorbar(image, ax=ax, label="u(x,t)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diffusion", choices=("ldg", "br2"), default="ldg")
    parser.add_argument("--quick", action="store_true", help="preset curto para teste")
    parser.add_argument("--K", type=int, default=None, help="número de elementos")
    parser.add_argument("--N", type=int, default=None, help="ordem polinomial")
    parser.add_argument("--t-end", type=float, default=None, help="tempo final")
    parser.add_argument("--dt", type=float, default=None, help="passo de tempo")
    parser.add_argument("--steps", type=int, default=None, help="número de passos")
    parser.add_argument("--print-every", type=int, default=0,
                        help="imprime progresso a cada n passos")
    parser.add_argument("--history-every", type=int, default=None,
                        help="guarda um frame a cada n passos")
    parser.add_argument("--output", type=Path, default=None, help="arquivo NPZ de saída")
    parser.add_argument("--no-plot", action="store_true", help="não gera PNG")
    args = parser.parse_args(argv)

    if args.quick:
        K, N, t_end, dt, default_steps = 10, 2, 0.005, 5.0e-4, 10
    else:
        # Effective production values in Alberto's executed Allen--Cahn loop.
        K, N, t_end, dt, default_steps = 20, 3, 5.0, 2.5e-4, 20_000
    K = args.K if args.K is not None else K
    N = args.N if args.N is not None else N
    t_end = args.t_end if args.t_end is not None else t_end
    dt = args.dt if args.dt is not None else dt
    if args.steps is not None:
        steps = args.steps
    elif args.t_end is not None or args.dt is not None:
        steps = int(math.ceil(t_end / dt))
    else:
        steps = default_steps
    if K < 1 or N < 0 or t_end <= 0.0 or dt <= 0.0 or steps < 1:
        raise ValueError("K>=1, N>=0, t-end>0, dt>0 e steps>=1 são necessários.")
    history_every = args.history_every
    if history_every is None:
        history_every = max(1, int(math.ceil(steps / 200)))
    if history_every < 1:
        raise ValueError("history-every deve ser positivo.")

    mu, beta, eps = 1.0, 0.5, math.sqrt(6.0)
    space = DGSpace1D(
        K, N, 0.0, 4.0 * math.pi, quad_type="GL", P=2 * N, nlP=4 * N
    )
    initial = project(space, np.cos(0.5 * space.xc))

    def rhs(states, time):
        del time
        u = states[0]
        padded = pad_periodic(space, u)

        # The source writes f_h=-beta*u and then applies -mu*d_x(f_h).
        # Keeping the projection explicit preserves the same anti-aliasing path.
        reaction_flux = FluxProjection(
            space, [padded], lambda value: -beta * value
        )[0]
        reaction = -mu * reaction_flux[:, 1:-1]

        if args.diffusion == "ldg":
            epst = np.full((space.nip, space.K + 2), mu * eps**2)
            epsb = np.full((2, space.K + 2), mu * eps**2)
            laplace = diffusion_rhs(
                space, padded, epst, epsb, method="ldg", periodic=True
            )
        else:
            laplace = diffusion_rhs(
                space, padded, method="br2", coefficient=mu * eps**2,
                eta=2.0, periodic=True,
            )
        return [reaction + laplace]

    history = History(space, history_every)
    state, time, actual_steps = integrate(
        [initial], rhs, t_end=t_end, dt=dt, steps=steps,
        history=history, print_every=max(args.print_every, 0),
    )
    x = space.xc.T.reshape(-1)
    final = history.physical(state[0])
    exact = np.exp(-time) * np.cos(0.5 * x)
    error = float(np.max(np.abs(final - exact)))

    output = args.output or (
        Path(__file__).resolve().parent / "results" / "generated"
        / f"allen_cahn_{args.diffusion}.npz"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        x=x,
        time=time,
        history_time=np.asarray(history.times),
        u=final,
        u_initial=history.values[0],
        u_history=history.array(),
        exact=exact,
        error_inf=error,
        K=K,
        N=N,
        dt=dt,
        steps=actual_steps,
        diffusion=args.diffusion,
    )

    final_png = output.with_name(output.stem + "_final.png")
    evolution_png = output.with_name(output.stem + "_evolution.png")
   
    if not args.no_plot:
        plot_final(final_png, x, history.values[0], final, exact, args.diffusion, time)
        plot_evolution(
            evolution_png, x, np.asarray(history.times), history.array(), args.diffusion
        )
    
    finite = bool(np.isfinite(final).all())
    print(
        f"Allen-Cahn ({args.diffusion.upper()}): t={time:.6g}, passos={actual_steps}, "
        f"finito={finite}, u=[{final.min():.6g},{final.max():.6g}]"
    )
    print(f"  erro_infinito={error:.6e}; resultado={output}")
    if not args.no_plot:
        print(f"  figuras={final_png}, {evolution_png}")
    if not finite:
        raise FloatingPointError("Allen-Cahn: estado não finito")


if __name__ == "__main__":
    main()
