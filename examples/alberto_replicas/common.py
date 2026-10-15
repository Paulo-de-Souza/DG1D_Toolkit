"""Shared assembly helpers for the Alberto test replicas.

The examples deliberately keep the same moving parts as the source scripts:
modal projection, padded modal states, Lax--Friedrichs/Rusanov interfaces,
SSP-RK time stepping, modal sensors and artificial viscosity.  The module is
only a small amount of plumbing; each example still exposes its own PDE,
initial data and boundary states.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

from dg1d_toolkit.dg1d_capture_shock import artificial_viscosity
from dg1d_toolkit.dg1d_core import DGSpace1D
from dg1d_toolkit.dg1d_integrators import RK4_Step, RKSSP104_Step, RKSSP54_Step
from dg1d_toolkit.dg1d_subcell import (
    diffusion_rhs,
    ldg_diffusion_rhs_many,
    pad_copy_boundary,
    pad_periodic,
)
from dg1d_toolkit.operators import FluxProjection
from dg1d_toolkit.numericalfluxes import rusanov


def add_runtime_options(parser):
    """Add the common resolution, time, progress and history controls."""
    parser.add_argument("--K", type=int, default=None, help="número de elementos")
    parser.add_argument("--N", type=int, default=None, help="ordem polinomial")
    parser.add_argument("--t-end", type=float, default=None, help="tempo final")
    parser.add_argument("--dt", type=float, default=None, help="passo de tempo")
    parser.add_argument("--steps", type=int, default=None,
                        help="número efetivo de passos; replica exatamente os scripts do Alberto")
    parser.add_argument("--print-every", type=int, default=0,
                        help="imprime progresso a cada n passos; 0 desativa")
    parser.add_argument("--history-every", type=int, default=0,
                        help="guarda um snapshot a cada n passos; 0 guarda apenas o final")
    parser.add_argument("--no-plot", action="store_true", help="não gera PNG")
    return parser


def apply_runtime_overrides(args, K, N, t_end, dt, steps=None):
    """Apply common CLI overrides after a case selects its preset."""
    K = args.K if args.K is not None else K
    N = args.N if args.N is not None else N
    t_end = args.t_end if args.t_end is not None else t_end
    dt = args.dt if args.dt is not None else dt
    if args.steps is not None:
        steps = args.steps
    elif steps is None:
        steps = int(np.ceil(t_end / dt))
    if K < 1 or N < 0 or t_end <= 0.0 or dt <= 0.0 or steps < 1:
        raise ValueError("K>=1, N>=0, t-end>0 e dt>0 são necessários.")
    return K, N, t_end, dt, steps


class StateHistory:
    """Collect selected modal states for evolution plots."""

    def __init__(self, space: DGSpace1D, every: int = 0):
        self.space = space
        self.every = max(int(every), 1)
        self.times = []
        self._states = []

    def __call__(self, state, time, step=0):
        if step == 0 or step % self.every == 0:
            self.times.append(float(time))
            self._states.append([np.array(value, copy=True) for value in state])

    def finalize(self, state, time, step):
        """Force the final state into the history if it was not on the grid."""
        if not self.times or abs(self.times[-1] - time) > 1.0e-14:
            self.times.append(float(time))
            self._states.append([np.array(value, copy=True) for value in state])

    def physical_arrays(self):
        if not self._states:
            return []
        return [
            np.stack([np.dot(self.space.psi, snapshot[index]).T.reshape(-1)
                      for snapshot in self._states])
            for index in range(len(self._states[0]))
        ]


def project(space: DGSpace1D, values: np.ndarray) -> np.ndarray:
    """L2-project values to modal coefficients.

    Both the standard and over-integration quadratures are accepted.  This is
    important for the nonlinear projections in the Alberto programs, where
    ``nlpsi``/``nlwi`` are used instead of the solution quadrature.
    """
    values = np.asarray(values, dtype=float)
    if values.shape == (space.nip, space.K):
        basis, weights = space.psi, space.wi
    elif values.shape == (space.nlnip, space.K):
        basis, weights = space.nlpsi, space.nlwi
    else:
        raise ValueError(
            "values deve ter shape "
            f"{(space.nip, space.K)} ou {(space.nlnip, space.K)}"
        )
    rhs = np.dot(weights[None, :] * basis.T, values)
    return space.InvM[:, None] * rhs


def padded(space: DGSpace1D, modal: np.ndarray, *, periodic: bool = False) -> np.ndarray:
    """Put a physical modal field into the toolkit's ``K+2`` convention."""
    return pad_periodic(space, modal) if periodic else pad_copy_boundary(space, modal)


def dirichlet_padded(
    space: DGSpace1D,
    modal: np.ndarray,
    left_value: float,
    right_value: float,
) -> np.ndarray:
    """Create ghost cells using Alberto's mean-reflection Dirichlet rule."""
    result = np.zeros((space.Nldof, space.K + 2), dtype=float)
    result[:, 1:-1] = modal
    left_trace = np.dot(space.Flkp1[0, :], modal[:, 0])
    right_trace = np.dot(space.Frk[0, :], modal[:, -1])
    result[0, 0] = -left_trace + 2.0 * left_value
    result[0, -1] = -right_trace + 2.0 * right_value
    return result


def inject_boundary_flux(space, state_pad, flux_pad, flux_function):
    """Set the boundary flux means after ``FluxProjection``."""
    left_state = np.dot(space.Flkp1[0, :], state_pad[:, 0])
    right_state = np.dot(space.Frk[0, :], state_pad[:, -1])
    flux_pad[0, 0] = float(np.asarray(flux_function(left_state)))
    flux_pad[0, -1] = float(np.asarray(flux_function(right_state)))
    return flux_pad


def interface_speeds(space: DGSpace1D, modal: np.ndarray, speed_function: Callable):
    """Build one local propagation speed for every mesh interface."""
    nodal = np.dot(space.psi, modal)
    speed = np.abs(np.asarray(speed_function(nodal), dtype=float))
    if speed.ndim == 0:
        speed = np.full((space.nip, space.K), float(speed))
    left = speed[0, :]
    right = speed[-1, :]
    result = np.empty(space.K + 1, dtype=float)
    result[0] = left[0]
    result[-1] = right[-1]
    if space.K > 1:
        result[1:-1] = np.maximum(right[:-1], left[1:])
    return result


def _convective_rhs(space, state_pad, flux_pad, speed):
    volume = np.dot(space.S.T, flux_pad[:, 1:-1])
    numerical = rusanov(space, state_pad, flux_pad, speed)
    return space.InvM[:, None] * space.J[None, :] ** (-1) * (volume + numerical)


def scalar_rhs(
    space: DGSpace1D,
    modal: np.ndarray,
    flux_function: Callable,
    speed_function: Callable,
    *,
    boundary_values: tuple[float, float] | None = None,
    artificial: bool = False,
    sensor_constant: float = 4.0,
    viscosity_factor: float = 0.5,
    profile: str = "linear",
    diffusion: str = "ldg",
):
    """DG right-hand side for a scalar conservation law plus optional AV."""
    periodic = boundary_values is None
    if periodic:
        state_pad = padded(space, modal, periodic=True)
    else:
        state_pad = dirichlet_padded(space, modal, *boundary_values)

    flux_pad = FluxProjection(space, [state_pad], flux_function)[0]
    inject_boundary_flux(space, state_pad, flux_pad, flux_function)
    speeds = interface_speeds(space, modal, speed_function)
    rhs = _convective_rhs(space, state_pad, flux_pad, speeds)

    if artificial:
        sensor, epst, epsb = artificial_viscosity(
            space,
            modal,
            wave_speed=speed_function,
            sensor="persson",
            periodic=periodic,
            profile=profile,
            sensor_constant=sensor_constant,
            viscosity_factor=viscosity_factor,
        )
        # Keep supplied ghost values for non-periodic Dirichlet tests.  The
        # updated dg1d_subcell operator intentionally preserves padded input.
        rhs = rhs + diffusion_rhs(
            space,
            state_pad,
            epst,
            epsb,
            method=diffusion,
            periodic=periodic,
        )
        return rhs, sensor
    return rhs, None


def euler_flux_safe(rho, rhou, energy, gamma=1.4):
    """Euler flux with floors used only to keep a diagnostic RK stage valid."""
    rho_safe = np.maximum(np.asarray(rho, dtype=float), 1.0e-10)
    kinetic = 0.5 * np.asarray(rhou, dtype=float) ** 2 / rho_safe
    pressure = np.maximum((gamma - 1.0) * (np.asarray(energy) - kinetic), 1.0e-10)
    velocity = np.asarray(rhou) / rho_safe
    return [
        np.asarray(rhou),
        np.asarray(rhou) ** 2 / rho_safe + pressure,
        (np.asarray(energy) + pressure) * velocity,
    ]


def euler_wave_speed(rho, rhou, energy, gamma=1.4):
    rho_safe = np.maximum(np.asarray(rho, dtype=float), 1.0e-10)
    kinetic = 0.5 * np.asarray(rhou, dtype=float) ** 2 / rho_safe
    pressure = np.maximum((gamma - 1.0) * (np.asarray(energy) - kinetic), 1.0e-10)
    return np.abs(np.asarray(rhou) / rho_safe) + np.sqrt(gamma * pressure / rho_safe)


def euler_rhs(
    space: DGSpace1D,
    state: Sequence[np.ndarray],
    left_state: Sequence[float],
    right_state: Sequence[float],
    *,
    gamma: float = 1.4,
    artificial: bool = True,
    sensor_constant: float = 10.0,
    viscosity_factor: float = 0.5,
    profile: str = "linear",
    diffusion: str = "ldg",
):
    """DG Euler RHS with the modal Persson/Warburton artificial viscosity."""
    if len(state) != 3:
        raise ValueError("Euler requer [rho, rhou, energy].")
    pads = []
    for modal, left, right in zip(state, left_state, right_state):
        pads.append(dirichlet_padded(space, modal, left, right))

    interior = [np.dot(space.psi, modal) for modal in state]
    rho_h, rhou_h, energy_h = interior
    speeds_h = euler_wave_speed(rho_h, rhou_h, energy_h, gamma)
    left = speeds_h[0, :]
    right = speeds_h[-1, :]
    speeds = np.empty(space.K + 1, dtype=float)
    speeds[0] = left[0]
    speeds[-1] = right[-1]
    if space.K > 1:
        speeds[1:-1] = np.maximum(right[:-1], left[1:])

    fluxes = FluxProjection(
        space,
        pads,
        lambda r, m, e: euler_flux_safe(r, m, e, gamma),
    )
    # FluxProjection returns one array per conserved component.  Explicitly
    # inject all three boundary flux means in the same conservative order.
    left_trace = [np.dot(space.Flkp1[0, :], p[:, 0]) for p in pads]
    right_trace = [np.dot(space.Frk[0, :], p[:, -1]) for p in pads]
    left_flux = euler_flux_safe(*left_trace, gamma)
    right_flux = euler_flux_safe(*right_trace, gamma)
    for j in range(3):
        fluxes[j][0, 0] = left_flux[j]
        fluxes[j][0, -1] = right_flux[j]

    rhs = [_convective_rhs(space, pads[j], fluxes[j], speeds) for j in range(3)]
    if artificial:
        sensor, epst, epsb = artificial_viscosity(
            space,
            state[0],
            wave_speed=lambda values: speeds_h,
            sensor="persson",
            periodic=False,
            profile=profile,
            sensor_constant=sensor_constant,
            viscosity_factor=viscosity_factor,
        )
        if diffusion.lower() == "ldg":
            # The artificial-viscosity matrices depend on epst/epsb, not on
            # the conserved component.  Assemble them once for rho, rhou and
            # energy instead of rebuilding them three times.
            diffusive_terms = ldg_diffusion_rhs_many(
                space, pads, epst, epsb, periodic=False
            )
        else:
            diffusive_terms = [
                diffusion_rhs(
                    space,
                    pads[j],
                    epst,
                    epsb,
                    method=diffusion,
                    periodic=False,
                )
                for j in range(3)
            ]
        rhs = [value + diffusive_terms[j] for j, value in enumerate(rhs)]
        return rhs, sensor
    return rhs, None


def integrate(
    state: Sequence[np.ndarray],
    rhs: Callable,
    *,
    t_end: float,
    dt: float,
    method: str = "rk54",
    limiter=None,
    diagnostics: Callable | None = None,
    progress_every: int = 0,
    progress_name: str = "DG",
    nsteps: int | None = None,
):
    """Advance a list of modal states with the toolkit integrator API."""
    methods = {
        "rk4": RK4_Step,
        "rk54": RKSSP54_Step,
        "rk104": RKSSP104_Step,
    }
    if method not in methods:
        raise ValueError(f"método temporal desconhecido: {method}")
    stepper = methods[method]
    current = [np.array(value, dtype=float, copy=True) for value in state]
    t = 0.0
    nsteps = int(nsteps if nsteps is not None else np.ceil(t_end / dt))
    def call_diagnostics(step):
        if diagnostics is None:
            return
        try:
            diagnostics(current, t, step)
        except TypeError:
            # Backward-compatible with two-argument callbacks.
            diagnostics(current, t)

    call_diagnostics(0)
    for step in range(1, nsteps + 1):
        h = min(dt, t_end - t)
        current = stepper(current, t, h, rhs, limiter)
        t += h
        call_diagnostics(step)
        if progress_every and (step == 1 or step % progress_every == 0 or step == nsteps):
            print(f"[{progress_name}] step {step:>7d}/{nsteps:<7d}  t={t:.8g}/{t_end:.8g}")
    return current, t, nsteps


def save_result(path: Path, space: DGSpace1D, state: Sequence[np.ndarray], **extra):
    """Save physical nodal fields and metadata in a small portable NPZ."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"x": space.xc.T.reshape(-1)}
    for index, modal in enumerate(state):
        data[f"u{index}"] = np.dot(space.psi, modal).T.reshape(-1)
    data.update(extra)
    np.savez(path, **data)


def plot_profiles(
    path: Path,
    space: DGSpace1D,
    state: Sequence[np.ndarray],
    labels: Sequence[str],
    title: str,
    *,
    initial_state: Sequence[np.ndarray] | None = None,
    references: Sequence[np.ndarray | None] | None = None,
    ylabels: Sequence[str] | None = None,
):
    """Save publication-friendly modal profiles without opening a GUI."""
    x = space.xc.T.reshape(-1)
    final_values = [np.dot(space.psi, modal).T.reshape(-1) for modal in state]
    initial_values = None
    if initial_state is not None:
        initial_values = [np.dot(space.psi, modal).T.reshape(-1) for modal in initial_state]
    return plot_arrays(path, x, final_values, labels, title,
                       initial_values=initial_values, references=references,
                       ylabels=ylabels)


def plot_arrays(
    path: Path,
    x: np.ndarray,
    final_values: Sequence[np.ndarray],
    labels: Sequence[str],
    title: str,
    *,
    initial_values: Sequence[np.ndarray] | None = None,
    references: Sequence[np.ndarray | None] | None = None,
    ylabels: Sequence[str] | None = None,
):
    """Save profiles that are already in physical space.

    This variant is used by Euler examples, whose plotted variables are
    derived fields (velocity and pressure) rather than the stored
    conservative modal coefficients.
    """
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    nplots = len(final_values)
    fig, axes = plt.subplots(nplots, 1, figsize=(8.2, 3.0 * nplots), squeeze=False)
    axes = axes[:, 0]

    for index, (axis, value, label) in enumerate(zip(axes, final_values, labels)):
        if initial_values is not None:
            axis.plot(x, initial_values[index], color="0.55", lw=1.2, ls="--", label="inicial")
        axis.plot(x, value, color="tab:blue", lw=1.6, label="DG final")
        if references is not None and references[index] is not None:
            axis.plot(x, np.asarray(references[index]).reshape(-1), color="tab:red", lw=1.2,
                      ls=":", label="referência")
        axis.set_ylabel(ylabels[index] if ylabels is not None else label)
        axis.grid(True, alpha=0.3)
        axis.legend(loc="best")
    axes[-1].set_xlabel("x")
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_evolution(path: Path, x, times, fields, labels, title):
    """Save one heat-map per field from a :class:`StateHistory`."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    nplots = len(fields)
    fig, axes = plt.subplots(nplots, 1, figsize=(8.2, 3.0 * nplots), squeeze=False)
    axes = axes[:, 0]
    for axis, field, label in zip(axes, fields, labels):
        image = axis.imshow(
            np.asarray(field),
            origin="lower",
            aspect="auto",
            extent=(float(np.min(x)), float(np.max(x)), float(times[0]), float(times[-1])),
            cmap="viridis",
        )
        axis.set_ylabel("t")
        axis.set_title(label)
        fig.colorbar(image, ax=axis, label=label)
        axis.grid(False)
    axes[-1].set_xlabel("x")
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def print_summary(name: str, space: DGSpace1D, state: Sequence[np.ndarray], t: float, steps: int):
    values = [np.dot(space.psi, modal) for modal in state]
    finite = all(np.isfinite(value).all() for value in values)
    ranges = ", ".join(f"u{i}=[{value.min():.4g},{value.max():.4g}]" for i, value in enumerate(values))
    print(f"{name}: t={t:.6g}, passos={steps}, finito={finite}, {ranges}")
    if not finite:
        raise FloatingPointError(f"{name}: estado não finito")
