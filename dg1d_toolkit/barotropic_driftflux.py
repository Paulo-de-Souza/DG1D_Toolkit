"""Barotropic drift-flux closures for a minimal 1-D two-phase system.

This numerical-study model sits between the scalar kinematic-wave example and
a full two-fluid model.  It has two phase mass balances and one mixture
momentum balance; it is not an industrial pressure-drop correlation.
"""

from __future__ import annotations

import numpy as np


class BarotropicDriftFlux:
    """Primitive recovery and fluxes for ``U = [m_l, m_g, G]``.

    Assumptions: constant liquid density, barotropic gas
    ``p = c_g**2 rho_g``, common pressure, algebraic slip
    ``u_g = C0*j + Vgj``, and no gravity/friction/energy/mass transfer.
    """

    @staticmethod
    def primitives(
        m_l: np.ndarray,
        m_g: np.ndarray,
        G: np.ndarray,
        rho_l: float,
        sound_speed_g: float,
        distribution_parameter: float = 1.05,
        drift_velocity: float = 0.08,
        alpha_floor: float = 1.0e-6,
        rho_g_floor: float = 1.0e-6,
        **kwargs,
    ) -> dict[str, np.ndarray]:
        r"""Recover ``alpha_l, alpha_g, rho_g, p, j, u_l, u_g`` from ``U``.

        The algebraic relations are

        .. math::

            G=m_lu_l+m_gu_g, \quad u_g=C_0j+V_{gj}, \quad
            j=\alpha_lu_l+\alpha_gu_g.

        Therefore no nonlinear primitive-variable solve is needed.
        """
        m_l_safe = np.clip(m_l, alpha_floor * rho_l, (1.0 - alpha_floor) * rho_l)
        alpha_l = m_l_safe / rho_l
        alpha_g = 1.0 - alpha_l
        rho_g = np.maximum(m_g / alpha_g, rho_g_floor)

        denominator = rho_l + alpha_g * (rho_g - rho_l) * distribution_parameter
        denominator = np.maximum(denominator, alpha_floor * rho_l)
        j = (G - alpha_g * (rho_g - rho_l) * drift_velocity) / denominator
        u_g = distribution_parameter * j + drift_velocity
        u_l = (j - alpha_g * u_g) / alpha_l
        pressure = sound_speed_g**2 * rho_g

        return {
            "alpha_l": alpha_l,
            "alpha_g": alpha_g,
            "rho_g": rho_g,
            "pressure": pressure,
            "j": j,
            "u_l": u_l,
            "u_g": u_g,
        }

    @classmethod
    def flux(
        cls,
        m_l: np.ndarray,
        m_g: np.ndarray,
        G: np.ndarray,
        **kwargs,
    ) -> list[np.ndarray]:
        r"""Return

        .. math::

            F(U)=[m_lu_l,\;m_gu_g,\;m_lu_l^2+m_gu_g^2+p]^{\mathsf T}.
        """
        q = cls.primitives(m_l, m_g, G, **kwargs)
        return [
            m_l * q["u_l"],
            m_g * q["u_g"],
            m_l * q["u_l"]**2 + m_g * q["u_g"]**2 + q["pressure"],
        ]

    @classmethod
    def conservative_from_j(
        cls,
        alpha_g: np.ndarray,
        rho_g: np.ndarray | float,
        mixture_velocity: np.ndarray | float,
        rho_l: float,
        distribution_parameter: float = 1.05,
        drift_velocity: float = 0.08,
        **kwargs,
    ) -> list[np.ndarray]:
        """Build a conservative state consistent with a prescribed ``j``."""
        alpha_l = 1.0 - alpha_g
        u_g = distribution_parameter * mixture_velocity + drift_velocity
        u_l = (mixture_velocity - alpha_g * u_g) / alpha_l
        m_l = rho_l * alpha_l
        m_g = rho_g * alpha_g
        G = m_l * u_l + m_g * u_g
        return [m_l, m_g, G]

    @classmethod
    def spectral_radius(
        cls,
        m_l: np.ndarray,
        m_g: np.ndarray,
        G: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Numerically evaluate the spectral radius of ``dF/dU``.

        This vectorized finite-difference Jacobian provides the local Rusanov
        speed without hard-coding a fragile eigenvalue formula for the first
        reduced closure.
        """
        state = [m_l, m_g, G]
        f0 = np.stack(cls.flux(*state, **kwargs), axis=-1)
        jacobian = np.empty(f0.shape[:-1] + (3, 3), dtype=float)
        reference_scales = (1.0, 0.10, 0.30)

        for column, (component, scale) in enumerate(zip(state, reference_scales)):
            delta = 1.0e-7 * np.maximum(np.abs(component), scale)
            perturbed = [item.copy() for item in state]
            perturbed[column] = perturbed[column] + delta
            f_perturbed = np.stack(cls.flux(*perturbed, **kwargs), axis=-1)
            jacobian[..., :, column] = (f_perturbed - f0) / delta[..., None]

        eigenvalues = np.linalg.eigvals(jacobian)
        return np.max(np.abs(eigenvalues), axis=-1)
