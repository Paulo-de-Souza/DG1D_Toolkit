"""Shock sensors and artificial-viscosity helpers for ``DGSpace1D``.

The implementation is a modular version of the active routines found in
Alberto's modal scripts: ``ShockIndicator`` and ``epssmoother``.  The sensor
is the modal Persson/Warburton indicator, while the viscosity is smoothed at
interfaces with the maximum of the adjacent cell values before being mapped
to the quadrature points of each element.

The functions deliberately return ordinary NumPy arrays so that they can be
inserted into the existing ``Lh`` functions without changing the integrators
or the limiter API.
"""

from dataclasses import dataclass
from typing import Callable, Optional, Union

import numpy as np


ArrayLike = Union[float, np.ndarray]


@dataclass
class ShockSensorResult:
    """Diagnostics and viscosity produced by :func:`persson_sensor`."""

    indicator: np.ndarray
    log_indicator: np.ndarray
    activation: np.ndarray
    viscosity: np.ndarray
    flagged: np.ndarray


def _cell_speed(space, u_modal, wave_speed: Optional[Union[ArrayLike, Callable]]):
    """Normalize a wave-speed specification to one value per cell."""
    if wave_speed is None:
        values = np.dot(space.psi, u_modal)
        return np.max(np.abs(values), axis=0)

    if callable(wave_speed):
        values = np.dot(space.psi, u_modal)
        wave_speed = wave_speed(values)

    speed = np.asarray(wave_speed, dtype=float)
    if speed.ndim == 0:
        return np.full(space.K, abs(float(speed)))
    if speed.shape == (space.nip, space.K):
        return np.max(np.abs(speed), axis=0)
    if speed.shape == (space.K,):
        return np.abs(speed)
    raise ValueError(
        "wave_speed deve ser escalar, (K,), (nip,K) ou uma função da solução nodal."
    )


def persson_sensor(
    space,
    u_modal: np.ndarray,
    wave_speed: Optional[Union[ArrayLike, Callable]] = None,
    *,
    s0: float = -2.0,
    kappa: float = 1.0,
    sensor_constant: float = 4.0,
    viscosity_factor: float = 0.5,
    floor: float = 1.0e-15,
) -> ShockSensorResult:
    """Compute a modal shock indicator and elementwise viscosity.

    Parameters mirror the values used in Alberto's Euler and Burgers shock
    capturing scripts.  ``indicator`` is the high-mode energy ratio ``F_k``;
    ``log_indicator`` is ``log10(min(C N^4 F_k, 1))``; ``activation`` is the
    smooth transition from zero to one; and ``viscosity`` is the dimensional
    coefficient

    ``nu_k = viscosity_factor * a_k * h_k / N * activation_k``.

    ``u_modal`` must have shape ``(N+1, K)`` without ghost cells.
    """
    u_modal = np.asarray(u_modal, dtype=float)
    expected = (space.Nldof, space.K)
    if u_modal.shape != expected:
        raise ValueError(f"u_modal deve ter shape {expected}; recebido {u_modal.shape}.")

    mass = np.asarray(space.diagM, dtype=float)
    energy = np.sum(mass[:, None] * u_modal**2, axis=0)
    high_mode_energy = mass[-1] * u_modal[-1, :] ** 2

    indicator = high_mode_energy / np.maximum(energy, floor)
    scaled = np.minimum(sensor_constant * max(space.N, 1) ** 4 * indicator, 1.0)
    log_indicator = np.log10(np.maximum(scaled, floor))

    activation = np.zeros(space.K, dtype=float)
    transition = (log_indicator >= s0 - kappa) & (log_indicator <= s0 + kappa)
    activation[log_indicator > s0 + kappa] = 1.0
    activation[transition] = 0.5 * (
        1.0 + np.sin(np.pi * (log_indicator[transition] - s0) / (2.0 * kappa))
    )

    speed = _cell_speed(space, u_modal, wave_speed)
    cell_size = 2.0 * np.asarray(space.J, dtype=float)
    degree = max(space.N, 1)
    maximum_viscosity = viscosity_factor * speed * cell_size / degree
    viscosity = maximum_viscosity * activation

    return ShockSensorResult(
        indicator=indicator,
        log_indicator=log_indicator,
        activation=activation,
        viscosity=viscosity,
        flagged=activation > 0.0,
    )


def concentration_sensor(
    space,
    u_modal: np.ndarray,
    wave_speed: Optional[Union[ArrayLike, Callable]] = None,
    *,
    threshold: float = 0.2,
    viscosity_factor: float = 0.5,
) -> np.ndarray:
    """Return Alberto's concentration-kernel cell detector.

    The modal derivative kernel is built from the same ``psi`` and ``Dhj``
    matrices already stored by ``DGSpace1D``.  The output is a cellwise
    viscosity: it is either zero or the local maximum viscosity scale.
    """
    u_modal = np.asarray(u_modal, dtype=float)
    if u_modal.shape != (space.Nldof, space.K):
        raise ValueError("u_modal deve conter somente os K elementos físicos.")

    # The source scripts use C = (pi/Nk)*sqrt(1-xi^2)*Dpsi.T as the kernel.
    nk = space.nip
    c_matrix = (
        (np.pi / nk)
        * np.sqrt(np.maximum(1.0 - space.xi**2, 0.0))[:, None]
        * space.Dpsi
    )
    c_matrix *= (np.pi / nk) * np.arange(1, nk + 1, dtype=float)[:, None]
    kernel = np.dot(c_matrix, u_modal)
    triggered = (nk * np.abs(kernel) ** 2 > threshold).any(axis=0)

    speed = _cell_speed(space, u_modal, wave_speed)
    local_max = viscosity_factor * speed * (2.0 * space.J) / max(space.N, 1)
    return np.where(triggered, local_max, 0.0)


def smooth_viscosity(
    space,
    cell_viscosity: np.ndarray,
    *,
    periodic: bool = False,
    profile: str = "linear",
    compact_order: int = 20,
    epsilon_machine: float = 1.0e-16,
):
    """Map cell viscosity to ``(epst, epsb)`` using Alberto's smoother.

    Interior face values are the maximum of the two neighboring cell values,
    exactly as in ``epssmoother``.  The linear profile is the default and is
    compatible with the toolkit's BR1 and LDG fluxes.  ``profile='compact'``
    additionally damps the coefficient inside each element, following the
    compact profile used by ``epscompactsmoother``.
    """
    cell_viscosity = np.asarray(cell_viscosity, dtype=float)
    if cell_viscosity.shape != (space.K,):
        raise ValueError(f"cell_viscosity deve ter shape ({space.K},).")
    if np.any(cell_viscosity < 0.0):
        raise ValueError("A viscosidade artificial deve ser não negativa.")
    if profile not in {"linear", "compact"}:
        raise ValueError("profile deve ser 'linear' ou 'compact'.")

    faces = np.zeros(space.K + 1, dtype=float)
    if periodic:
        faces[0] = max(cell_viscosity[-1], cell_viscosity[0])
        faces[-1] = faces[0]
    else:
        faces[0] = cell_viscosity[0]
        faces[-1] = cell_viscosity[-1]
    if space.K > 1:
        faces[1:-1] = np.maximum(cell_viscosity[:-1], cell_viscosity[1:])

    epst = np.zeros((space.nip, space.K + 2), dtype=float)
    xi = np.asarray(space.xi)[:, None]
    left = faces[:-1][None, :]
    right = faces[1:][None, :]
    epst[:, 1:-1] = 0.5 * (right - left) * (xi + 1.0) + left

    if profile == "compact":
        alpha = -np.log(epsilon_machine)
        taper = np.exp(-alpha * np.abs(space.xi)[:, None] ** (2 * compact_order))
        epst[:, 1:-1] *= taper

    # Ghost coefficients follow the nearest physical element, as in the
    # source scripts.  Face values are kept separately for LDG/BR2 assembly.
    epst[:, 0] = epst[:, 1]
    epst[:, -1] = epst[:, -2]

    epsb = np.zeros((2, space.K + 2), dtype=float)
    epsb[:, 0] = faces[0]
    epsb[:, -1] = faces[-1]
    epsb[0, 1:-1] = faces[:-1]
    epsb[1, 1:-1] = faces[1:]
    return epst, epsb


def artificial_viscosity(
    space,
    u_modal: np.ndarray,
    wave_speed: Optional[Union[ArrayLike, Callable]] = None,
    *,
    sensor: str = "persson",
    periodic: bool = False,
    profile: str = "linear",
    **sensor_kwargs,
):
    """Compute cell, quadrature and face artificial-viscosity arrays.

    Returns ``(sensor_result, epst, epsb)`` for the Persson/Warburton sensor,
    or ``(None, epst, epsb)`` for the concentration detector.  The tuple is
    intentionally convenient inside a right-hand-side function.
    """
    if sensor == "persson":
        result = persson_sensor(space, u_modal, wave_speed, **sensor_kwargs)
        cell_viscosity = result.viscosity
    elif sensor == "concentration":
        result = None
        cell_viscosity = concentration_sensor(space, u_modal, wave_speed, **sensor_kwargs)
    else:
        raise ValueError("sensor deve ser 'persson' ou 'concentration'.")

    epst, epsb = smooth_viscosity(
        space,
        cell_viscosity,
        periodic=periodic,
        profile=profile,
    )
    return result, epst, epsb
