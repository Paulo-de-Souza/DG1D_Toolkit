"""Numerical fluxes used by the DG1D toolkit.

This file is a drop-in replacement for ``dg1d_toolkit.numericalfluxes``.
The original Lax--Friedrichs, Rusanov and BR1 functions are preserved.  The
additional LDG and BR2 helpers follow the formulations used in Alberto's
modal DG scripts:

* LDG uses the Hesthaven--Warburton one-sided lifting with the arithmetic
  average of ``sqrt(epsilon)`` at an interface;
* BR2 exposes the lifting of the solution jump and the resulting stabilization
  contribution.  The complete scalar diffusion operator is assembled in
  :mod:`dg1d_toolkit.dg1d_subcell`.

Arrays follow the convention already used by the toolkit: modal fields have
shape ``(N+1, K+2)`` and the first/last columns are ghost cells.  A viscosity
field ``epst`` has shape ``(nip, K+2)``; ``epsb`` has shape ``(2, K+2)`` and
stores left/right face values.
"""

import numpy as np


def lax_friedrichs(space, ut, ft, max_speed=None):
    """Calculate Lax--Friedrichs interface contributions."""
    if max_speed is None:
        max_speed = 1

    flux_num = (
        -0.5 * np.dot(space.Flkp1, (ft[:, 2:] - max_speed * ut[:, 2:]))
        -0.5 * np.dot(space.Frk, (ft[:, 1:-1] + max_speed * ut[:, 1:-1]))
        +0.5 * np.dot(space.Frkm1, (ft[:, :-2] + max_speed * ut[:, :-2]))
        +0.5 * np.dot(space.Flk, (ft[:, 1:-1] - max_speed * ut[:, 1:-1]))
    )

    return flux_num


def br1_flux(space, field, epst):
    """Calculate BR1 central interface contributions.

    ``field`` can be the padded solution used to compute ``q`` or the padded
    auxiliary field ``q`` used by the second diffusion application.
    """
    e_C = np.sqrt(np.maximum(epst[0, 1:-1], 0.0))
    e_R = np.sqrt(np.maximum(epst[0, 2:], 0.0))
    e_L = np.sqrt(np.maximum(epst[0, :-2], 0.0))

    flux_num = (
        +0.5 * (e_R * np.dot(space.Flkp1, field[:, 2:])
                 + e_C * np.dot(space.Frk, field[:, 1:-1]))
        -0.5 * (e_L * np.dot(space.Frkm1, field[:, :-2])
                 + e_C * np.dot(space.Flk, field[:, 1:-1]))
    )

    return flux_num


def rusanov(space, ut, ft, C_local):
    """Calculate Rusanov/local Lax--Friedrichs contributions."""
    C_local = np.asarray(C_local)
    flux_num = (
        -0.5 * (np.dot(space.Flkp1, ft[:, 2:])
                - C_local[1:] * np.dot(space.Flkp1, ut[:, 2:]))
        -0.5 * (np.dot(space.Frk, ft[:, 1:-1])
                + C_local[1:] * np.dot(space.Frk, ut[:, 1:-1]))
        +0.5 * (np.dot(space.Frkm1, ft[:, :-2])
                + C_local[:-1] * np.dot(space.Frkm1, ut[:, :-2]))
        +0.5 * (np.dot(space.Flk, ft[:, 1:-1])
                - C_local[:-1] * np.dot(space.Flk, ut[:, 1:-1]))
    )
    return flux_num


def ldg_flux(space, field, epsb):
    """LDG gradient flux for the Hesthaven--Warburton variant.

    This is the interface part used when computing ``q = d(sqrt(eps) u)/dx``.
    It corresponds to the active LDG branch in Alberto's shock-capturing
    scripts, where neighboring face values of ``sqrt(eps)`` are averaged.
    """
    e_right = (
        np.sqrt(np.maximum(epsb[0, 2:], 0.0))
        + np.sqrt(np.maximum(epsb[-1, 1:-1], 0.0))
    )
    e_left = (
        np.sqrt(np.maximum(epsb[-1, :-2], 0.0))
        + np.sqrt(np.maximum(epsb[0, 1:-1], 0.0))
    )

    return (
        0.5 * e_right * np.dot(space.Flkp1, field[:, 2:])
        - 0.5 * e_left * np.dot(space.Flk, field[:, 1:-1])
    )


def ldg_diffusive_flux(space, field, epsb):
    """LDG one-sided flux used in the second diffusion application."""
    e_center_right = np.sqrt(np.maximum(epsb[-1, 1:-1], 0.0))
    e_left_right = np.sqrt(np.maximum(epsb[-1, :-2], 0.0))

    return (
        e_center_right * np.dot(space.Frk, field[:, 1:-1])
        - e_left_right * np.dot(space.Frkm1, field[:, :-2])
    )


def br2_liftings(space, field):
    """Return BR2 liftings of solution jumps at cell faces.

    The returned pair ``(R_left, R_right)`` has the same padded shape as
    ``field``.  ``R_left[:, k]`` is the lifting associated with the left face
    of cell ``k`` and ``R_right[:, k]`` with its right face.  The sign and
    factor ``1/2`` match the BR2 construction in Alberto's
    ``Allen-Cahn-1D-Modal-DG-BR2-v0.py``.
    """
    interior = field[:, 1:-1]
    left = field[:, :-2]
    right = field[:, 2:]

    R_left = np.zeros_like(field, dtype=float)
    R_right = np.zeros_like(field, dtype=float)

    # For the periodic and reflective examples the caller fills ghost cells
    # before reaching this routine.  The same padded arrays also make the
    # boundary contribution explicit and easy to override.
    R_left[:, 1:-1] = -0.5 * (
        space.InvM[:, None] * space.J[None, :] ** (-1)
        * (np.dot(space.Flk, interior) - np.dot(space.Frkm1, left))
    )
    R_right[:, 1:-1] = -0.5 * (
        space.InvM[:, None] * space.J[None, :] ** (-1)
        * (np.dot(space.Frk, interior) - np.dot(space.Flkp1, right))
    )

    return R_left, R_right


def br2_flux(space, field, eta=2.0):
    """Return the BR2 lifting stabilization contribution.

    The result has shape ``(N+1, K)`` and is ready to be added to the local
    diffusion numerator.  The remaining volume and averaged-gradient terms
    are assembled by ``br2_diffusion_rhs`` in ``dg1d_subcell.py``.
    """
    R_left, R_right = br2_liftings(space, field)

    return 0.5 * eta * (
        np.dot(space.Flkp1, R_left[:, 2:])
        + np.dot(space.Frk, R_right[:, 1:-1])
        - np.dot(space.Frkm1, R_right[:, :-2])
        - np.dot(space.Flk, R_left[:, 1:-1])
    )
