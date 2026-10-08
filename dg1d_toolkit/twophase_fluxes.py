"""Physical closures used by the 1-D two-phase examples.

The routines here deliberately implement the *reduced* models used as a
first numerical verification step.  They are not replacements for a complete
two-fluid / ALFASim model, whose closures require phase momentum, pressure,
wall and interfacial shear, and thermodynamics.
"""

from __future__ import annotations

import numpy as np


class DriftFlux:
    """Closures for a scalar kinematic-wave drift-flux model."""

    @staticmethod
    def gas_void_fraction_flux(
        alpha_g: np.ndarray,
        mixture_velocity: float,
        distribution_parameter: float = 1.20,
        drift_velocity: float = 0.35,
        **kwargs,
    ) -> np.ndarray:
        r"""Return the gas volumetric flux for the reduced drift-flux model.

        The governing equation is

        .. math::

            \alpha_{g,t} + \partial_x F_g(\alpha_g) = 0,
            \qquad
            F_g = \alpha_g\,[C_0 j + V_{gj}(1-\alpha_g)].

        Here ``alpha_g`` is the gas void fraction, ``j`` is a prescribed
        mixture superficial velocity, ``C0`` is the distribution parameter,
        and ``Vgj`` is a prescribed drift velocity.  This closure is the
        one-equation kinematic-wave limit of a drift-flux model: it retains
        gas/liquid slip but freezes pressure, mixture momentum, energy,
        gravity, and friction.

        Parameters
        ----------
        alpha_g : ndarray
            Gas void fraction evaluated at quadrature points.  Expected range:
            ``0 <= alpha_g <= 1``.
        mixture_velocity : float
            Prescribed total superficial velocity :math:`j` [m/s].
        distribution_parameter : float, optional
            Drift-flux distribution parameter :math:`C_0` [-].
        drift_velocity : float, optional
            Gas drift velocity :math:`V_{gj}` [m/s].

        Returns
        -------
        ndarray
            Gas volumetric flux :math:`F_g` [m/s].
        """
        return alpha_g * (
            distribution_parameter * mixture_velocity
            + drift_velocity * (1.0 - alpha_g)
        )

    @staticmethod
    def gas_void_fraction_speed(
        alpha_g: np.ndarray,
        mixture_velocity: float,
        distribution_parameter: float = 1.20,
        drift_velocity: float = 0.35,
        **kwargs,
    ) -> np.ndarray:
        r"""Return :math:`dF_g/d\alpha_g`, used by the Rusanov flux/CFL."""
        return (
            distribution_parameter * mixture_velocity
            + drift_velocity * (1.0 - 2.0 * alpha_g)
        )
