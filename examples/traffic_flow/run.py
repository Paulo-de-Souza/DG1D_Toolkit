"""Standalone traffic-flow Green-Light reproduction of Alberto's test.

The example is intentionally complete: it assembles the scalar conservation
law, the local Rusanov flux, Persson/Warburton artificial viscosity with LDG,
the SSP-RK54 time step, and all post-processing locally.  The production
preset is the effective configuration used by Alberto's executed loop.

Examples from the repository root::

    python examples/traffic_flow/run.py --quick
    python examples/traffic_flow/run.py --print-every 500 --history-every 20

The run writes an NPZ, the final profile, the space--time map, and a GIF to
``examples/traffic_flow/results/generated``.
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

from dg1d_toolkit.dg1d_capture_shock import artificial_viscosity
from dg1d_toolkit.dg1d_core import DGSpace1D
from dg1d_toolkit.dg1d_integrators import RKSSP54_Step
from dg1d_toolkit.dg1d_postprocessing import create_gif
from dg1d_toolkit.dg1d_subcell import diffusion_rhs
from dg1d_toolkit.numericalfluxes import rusanov
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


def dirichlet_padded(space: DGSpace1D, modal: np.ndarray,
                     left_value: float, right_value: float) -> np.ndarray:
    """Alberto's mean-reflection Dirichlet ghost-cell construction."""
    padded = np.zeros((space.Nldof, space.K + 2), dtype=float)
    padded[:, 1:-1] = modal
    left_trace = np.dot(space.Flkp1[0, :], modal[:, 0])
    right_trace = np.dot(space.Frk[0, :], modal[:, -1])
    padded[0, 0] = -left_trace + 2.0 * left_value
    padded[0, -1] = -right_trace + 2.0 * right_value
    return padded


def inject_boundary_flux(space: DGSpace1D, state_pad: np.ndarray,
                         flux_pad: np.ndarray, flux_function) -> np.ndarray:
    left_state = np.dot(space.Flkp1[0, :], state_pad[:, 0])
    right_state = np.dot(space.Frk[0, :], state_pad[:, -1])
    flux_pad[0, 0] = float(np.asarray(flux_function(left_state)))
    flux_pad[0, -1] = float(np.asarray(flux_function(right_state)))
    return flux_pad


def interface_speeds(space: DGSpace1D, modal: np.ndarray,
                     speed_function) -> np.ndarray:
    nodal = np.dot(space.psi, modal)
    speed = np.abs(np.asarray(speed_function(nodal), dtype=float))
    if speed.ndim == 0:
        speed = np.full((space.nip, space.K), float(speed))
    result = np.empty(space.K + 1, dtype=float)
    result[0] = speed[0, 0]
    result[-1] = speed[-1, -1]
    if space.K > 1:
        result[1:-1] = np.maximum(speed[-1, :-1], speed[0, 1:])
    return result


def scalar_rhs(space: DGSpace1D, modal: np.ndarray, flux_function,
               speed_function, left_value: float, right_value: float):
    """Convective DG residual plus Alberto's LDG artificial viscosity."""
    state_pad = dirichlet_padded(space, modal, left_value, right_value)
    flux_pad = FluxProjection(space, [state_pad], flux_function)[0]
    inject_boundary_flux(space, state_pad, flux_pad, flux_function)
    speeds = interface_speeds(space, modal, speed_function)
    volume = np.dot(space.S.T, flux_pad[:, 1:-1])
    numerical = rusanov(space, state_pad, flux_pad, speeds)
    convective = space.InvM[:, None] * space.J[None, :] ** (-1) * (
        volume + numerical
    )

    sensor, epst, epsb = artificial_viscosity(
        space,
        modal,
        wave_speed=speed_function,
        sensor="persson",
        periodic=False,
        profile="linear",
        sensor_constant=1.0,
        viscosity_factor=0.5,
    )
    diffusive = diffusion_rhs(
        space, state_pad, epst, epsb, method="ldg", periodic=False
    )
    return convective + diffusive, sensor


class History:
    """Store selected physical solution snapshots."""

    def __init__(self, space: DGSpace1D, every: int):
        self.space = space
        self.every = max(int(every), 1)
        self.times: list[float] = []
        self.values: list[np.ndarray] = []

    def physical(self, modal: np.ndarray) -> np.ndarray:
        return np.dot(self.space.psi, modal).T.reshape(-1)

    def record(self, state: list[np.ndarray], time: float, step: int) -> None:
        if step == 0 or step % self.every == 0:
            self.times.append(float(time))
            self.values.append(self.physical(state[0]))

    def finalize(self, state: list[np.ndarray], time: float) -> None:
        if not self.times or abs(self.times[-1] - time) > 1.0e-14:
            self.times.append(float(time))
            self.values.append(self.physical(state[0]))

    def array(self) -> np.ndarray:
        return np.stack(self.values, axis=0)


def integrate(initial: list[np.ndarray], rhs, *, t_end: float, dt: float,
              steps: int, history: History, print_every: int):
    state = [np.array(value, dtype=float, copy=True) for value in initial]
    time = 0.0
    actual_steps = 0
    history.record(state, time, 0)
    sensor = None
    for step in range(1, steps + 1):
        h = min(dt, max(t_end - time, 0.0))
        if h == 0.0:
            break

        def rhs_with_sensor(current, current_time):
            nonlocal sensor
            result, sensor = rhs(current, current_time)
            return [result]

        state = RKSSP54_Step(state, time, h, rhs_with_sensor)
        time += h
        actual_steps = step
        history.record(state, time, step)
        if print_every and (step == 1 or step % print_every == 0 or step == steps):
            print(f"[Traffic-Flow] step {step:>7d}/{steps:<7d}  t={time:.8g}/{t_end:.8g}")
    history.finalize(state, time)
    return state, time, actual_steps, sensor


def plot_final(path: Path, x: np.ndarray, initial: np.ndarray, final: np.ndarray,
               time: float) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.plot(x, initial, "--", color="0.45", lw=1.2, label="inicial")
    ax.plot(x, final, color="tab:blue", lw=1.6, label="DG final")
    ax.set_xlabel("x")
    ax.set_ylabel("densidade u(x,t)")
    ax.set_title(f"Traffic Flow — Green Light — t={time:g}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_evolution(path: Path, x: np.ndarray, times: np.ndarray,
                   values: np.ndarray) -> None:
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
    ax.set_title("Traffic Flow — mapa espaço–tempo")
    fig.colorbar(image, ax=ax, label="densidade u(x,t)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--no-gif", action="store_true", help="não gera GIF")
    args = parser.parse_args(argv)

    if args.quick:
        K, N, t_end, dt, default_steps = 12, 3, 0.01, 5.0e-4, 20
    else:
        # Effective production values in Alberto's executed traffic-flow loop.
        K, N, t_end, dt, default_steps = 39, 9, 1.0, 2.5e-4, 4_000
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
        return scalar_rhs(
            space, states[0], flux, speed, 0.0, 0.0
        )

    history = History(space, history_every)
    state, time, actual_steps, sensor = integrate(
        [initial], rhs, t_end=t_end, dt=dt, steps=steps,
        history=history, print_every=max(args.print_every, 0),
    )
    final = history.values[-1]
    x = space.xc.T.reshape(-1)

    # Re-evaluate the sensor on the actual final state for a reproducible
    # diagnostic, instead of relying on the last internal RK stage.
    sensor, _, _ = artificial_viscosity(
        space, state[0], wave_speed=speed, sensor="persson", periodic=False,
        profile="linear", sensor_constant=1.0, viscosity_factor=0.5,
    )
    output = args.output or (
        Path(__file__).resolve().parent / "results" / "generated" / "traffic_flow.npz"
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
        sensor_activation=sensor.activation,
        sensor_viscosity=sensor.viscosity,
        Vm=Vm,
        um=um,
        K=K,
        N=N,
        dt=dt,
        steps=actual_steps,
        sensor_constant=1.0,
        viscosity_factor=0.5,
        diffusion="ldg",
        boundary_values=np.asarray([0.0, 0.0]),
    )

    final_png = output.with_name(output.stem + "_final.png")
    evolution_png = output.with_name(output.stem + "_evolution.png")
    animation_dir = output.parent / (output.stem + "_animation")
    gif_path = animation_dir / (output.stem + ".gif")
    if not args.no_plot:
        plot_final(final_png, x, history.values[0], final, time)
        plot_evolution(
            evolution_png, x, np.asarray(history.times), history.array()
        )
    if not args.no_gif:
        frame_dt = float(np.mean(np.diff(history.times))) if len(history.times) > 1 else dt
        ylim = (float(history.array().min()), float(history.array().max()))
        span = max(ylim[1] - ylim[0], 1.0e-12)
        margin = 0.05 * span
        create_gif(
            x,
            history.array().T,
            frame_dt,
            folder=str(animation_dir),
            filename=gif_path.name,
            xlabel="x",
            ylabel="densidade u(x,t)",
            title="Traffic Flow — Green Light",
            xlim=(float(x.min()), float(x.max())),
            ylim=(ylim[0] - margin, ylim[1] + margin),
            duration=80,
        )

    finite = bool(np.isfinite(final).all())
    active = int(np.count_nonzero(sensor.flagged))
    print(
        f"Traffic-Flow: t={time:.6g}, passos={actual_steps}, finito={finite}, "
        f"u=[{final.min():.6g},{final.max():.6g}]"
    )
    print(f"  captura=Persson/Warburton + LDG; células ativas={active}/{K}")
    print(f"  resultado={output}")
    if not args.no_plot:
        print(f"  figuras={final_png}, {evolution_png}")
    if not args.no_gif:
        print(f"  gif={gif_path}")
    if not finite:
        raise FloatingPointError("Traffic-Flow: estado não finito")


if __name__ == "__main__":
    main()
