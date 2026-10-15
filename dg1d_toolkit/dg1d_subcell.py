"""Diffusive/shock-capturing operators built on the DG1D toolkit.

The Alberto scripts call their artificial-viscosity experiments
"sub-cell shock capturing".  The active mechanism there is a DG artificial
viscosity operator, not a separate finite-volume subcell time integrator.
This module preserves that mechanism and adds small subcell utilities for
diagnostics and an optional modal fallback.
"""

import numpy as np

from dg1d_toolkit.mathDG1D import JacobiP
from dg1d_toolkit.numericalfluxes import (
    br1_flux,
    br2_flux,
    ldg_diffusive_flux,
    ldg_flux,
)
from dg1d_toolkit.operators import ModifStiffMatrix


def pad_periodic(space, field):
    """Return a padded modal field with periodic ghost cells."""
    field = np.asarray(field, dtype=float)
    if field.shape == (space.Nldof, space.K + 2):
        padded = field.copy()
        physical = padded[:, 1:-1]
    elif field.shape == (space.Nldof, space.K):
        physical = field
        padded = np.zeros((space.Nldof, space.K + 2), dtype=float)
        padded[:, 1:-1] = physical
    else:
        raise ValueError(
            f"field deve ter shape {(space.Nldof, space.K)} ou "
            f"{(space.Nldof, space.K + 2)}."
        )

    padded[:, 0] = physical[:, -1]
    padded[:, -1] = physical[:, 0]
    return padded


def pad_copy_boundary(space, field):
    """Return a padded field with nearest-cell ghost values."""
    field = np.asarray(field, dtype=float)
    if field.shape == (space.Nldof, space.K + 2):
        padded = field.copy()
        padded[:, 0] = padded[:, 1]
        padded[:, -1] = padded[:, -2]
        return padded
    if field.shape != (space.Nldof, space.K):
        raise ValueError("field tem dimensão modal/celular incompatível com space.")
    padded = np.zeros((space.Nldof, space.K + 2), dtype=float)
    padded[:, 1:-1] = field
    padded[:, 0] = field[:, 0]
    padded[:, -1] = field[:, -1]
    return padded


def _prepare_padded(space, field, periodic):
    """Normalize a field without destroying caller-supplied ghost states.

    The original Alberto programs prescribe boundary data directly in the
    ghost modal means.  Keeping an already-padded array intact lets the
    modular operators reproduce that pattern; only an unpadded field is
    filled automatically.
    """
    field = np.asarray(field, dtype=float)
    if field.shape == (space.Nldof, space.K + 2):
        return field.copy()
    if periodic:
        return pad_periodic(space, field)
    return pad_copy_boundary(space, field)


def _periodicize_in_place(space, field):
    field[:, 0] = field[:, -2]
    field[:, -1] = field[:, 1]
    return field


def _unit_epst(space):
    epst = np.ones((space.nip, space.K + 2), dtype=float)
    epsb = np.ones((2, space.K + 2), dtype=float)
    return epst, epsb


def br1_diffusion_rhs(space, field, epst, *, periodic=True):
    """Assemble the BR1 scalar diffusion operator.

    This is the same two-pass ``q`` construction already used by the
    toolkit's viscous Burgers example, generalized to the padded field and a
    cell-dependent artificial viscosity.
    """
    field = _prepare_padded(space, field, periodic)
    epst = np.asarray(epst, dtype=float)
    if epst.shape != (space.nip, space.K + 2):
        raise ValueError("epst deve ter shape (space.nip, space.K+2).")

    interior = field[:, 1:-1]
    sqrt_eps = np.sqrt(np.maximum(epst[0, 1:-1], 0.0))
    q = np.zeros_like(field)
    q[:, 1:-1] = space.InvM[:, None] * space.J[None, :] ** (-1) * (
        -sqrt_eps[None, :] * np.dot(space.S.T, interior)
        + br1_flux(space, field, epst)
    )
    if periodic:
        _periodicize_in_place(space, q)
    else:
        q[:, 0] = q[:, 1]
        q[:, -1] = q[:, -2]

    diffusive_numerator = (
        -sqrt_eps[None, :] * np.dot(space.S.T, q[:, 1:-1])
        + br1_flux(space, q, epst)
    )
    return space.InvM[:, None] * space.J[None, :] ** (-1) * diffusive_numerator


def ldg_diffusion_rhs(space, field, epst, epsb, *, periodic=True):
    """Assemble the variable-coefficient LDG diffusion operator.

    ``ModifStiffMatrix(..., type_form='Warburton')`` is evaluated element by
    element, as in Alberto's active shock-capturing branch.  The two modified
    stiffness matrices are then paired with the LDG interface fluxes.
    """
    field = _prepare_padded(space, field, periodic)
    epst = np.asarray(epst, dtype=float)
    epsb = np.asarray(epsb, dtype=float)
    expected_epst = (space.nip, space.K + 2)
    expected_epsb = (2, space.K + 2)
    if epst.shape != expected_epst or epsb.shape != expected_epsb:
        raise ValueError("epst/epsb têm dimensão incompatível com space.")

    s_first = np.zeros((space.K, space.Nldof, space.Nldof), dtype=float)
    s_second = np.zeros_like(s_first)
    for k in range(space.K):
        s_first[k], s_second[k] = ModifStiffMatrix(
            space, epst[:, k + 1], epsb[:, k + 1], type_form="Warburton"
        )

    interior = field[:, 1:-1]
    second_volume = np.einsum("kij,jk->ik", s_second, interior)

    q = np.zeros_like(field)
    q[:, 1:-1] = space.InvM[:, None] * space.J[None, :] ** (-1) * (
        -second_volume + ldg_flux(space, field, epsb)
    )
    if periodic:
        _periodicize_in_place(space, q)
    else:
        q[:, 0] = q[:, 1]
        q[:, -1] = q[:, -2]

    second_q = np.einsum("kij,jk->ik", s_first, q[:, 1:-1])
    diffusive_numerator = -second_q + ldg_diffusive_flux(space, q, epsb)
    return space.InvM[:, None] * space.J[None, :] ** (-1) * diffusive_numerator


def _boundary_derivative_vectors(space):
    """Return modal derivatives of the basis at xi=-1 and xi=+1."""
    degrees = np.arange(space.Nldof, dtype=float)
    right = 0.5 * degrees * (degrees + 1.0)
    left = ((-1.0) ** np.maximum(degrees - 1.0, 0.0)) * right
    left[0] = 0.0
    return left, right


def br2_diffusion_rhs(space, field, *, coefficient: float = 1.0, eta: float = 2.0,
                      periodic=True):
    """Assemble the scalar constant-coefficient BR2 diffusion operator.

    The construction follows the active BR2 mixed formulation in Alberto's
    Allen--Cahn script: central ``q``, averaged boundary gradients, and the
    four local liftings of the neighboring solution jumps.  ``coefficient``
    is kept scalar because that is the BR2 formulation actually exercised in
    the source script; variable artificial viscosity is handled by BR1/LDG.
    """
    if coefficient < 0.0:
        raise ValueError("coefficient deve ser não negativo.")
    field = _prepare_padded(space, field, periodic)
    unit_epst, _ = _unit_epst(space)

    # Central BR1 gradient q for coefficient one.
    q = np.zeros_like(field)
    q[:, 1:-1] = space.InvM[:, None] * space.J[None, :] ** (-1) * (
        -np.dot(space.S.T, field[:, 1:-1])
        + br1_flux(space, field, unit_epst)
    )
    if periodic:
        _periodicize_in_place(space, q)
    else:
        q[:, 0] = q[:, 1]
        q[:, -1] = q[:, -2]

    left_d, right_d = _boundary_derivative_vectors(space)
    scale = np.asarray(space.J, dtype=float) ** (-1)
    grad_left = scale[None, :] * np.dot(left_d, field[:, 1:-1])[None, :]
    grad_right = scale[None, :] * np.dot(right_d, field[:, 1:-1])[None, :]

    # The boundary derivatives of the neighboring element at the same face.
    grad_right_neighbor_left = scale[None, :] * np.dot(left_d, field[:, 2:])[None, :]
    grad_left_neighbor_right = scale[None, :] * np.dot(right_d, field[:, :-2])[None, :]

    averaged_right = 0.5 * (grad_right + grad_right_neighbor_left)
    averaged_left = 0.5 * (grad_left + grad_left_neighbor_right)

    interface_terms = (
        np.dot(space.Frk, q[:, 1:-1]) - averaged_right
        - np.dot(space.Frkm1, q[:, :-2]) + averaged_left
    )
    stabilization = br2_flux(space, field, eta=eta)

    numerator = -np.dot(space.S.T, q[:, 1:-1]) + interface_terms + stabilization
    return coefficient * space.InvM[:, None] * space.J[None, :] ** (-1) * numerator


def diffusion_rhs(space, field, epst=None, epsb=None, *, method="br1",
                  coefficient=1.0, eta=2.0, periodic=True):
    """Dispatch to ``br1``, ``ldg`` or ``br2`` diffusion.

    For ``br1`` and ``ldg``, ``epst``/``epsb`` are required.  ``br2`` uses
    ``coefficient`` and the constant-coefficient formulation above.
    """
    method = method.lower()
    if method == "br1":
        if epst is None:
            raise ValueError("br1 requer epst.")
        return br1_diffusion_rhs(space, field, epst, periodic=periodic)
    if method == "ldg":
        if epst is None or epsb is None:
            raise ValueError("ldg requer epst e epsb.")
        return ldg_diffusion_rhs(space, field, epst, epsb, periodic=periodic)
    if method == "br2":
        return br2_diffusion_rhs(
            space, field, coefficient=coefficient, eta=eta, periodic=periodic
        )
    raise ValueError("method deve ser 'br1', 'ldg' ou 'br2'.")


def subcell_averages(space, u_modal, nsub=None):
    """Compute conservative averages on reference subcells.

    This is a diagnostic/reconstruction utility.  It does not claim to be a
    complete finite-volume subcell time stepper; the Alberto source material
    likewise uses artificial viscosity as its active shock-capture mechanism.
    """
    u_modal = np.asarray(u_modal, dtype=float)
    if u_modal.shape != (space.Nldof, space.K):
        raise ValueError("u_modal deve ter shape (N+1,K).")
    nsub = int(nsub or max(space.N + 1, 2))
    if nsub < 1:
        raise ValueError("nsub deve ser positivo.")

    qx, qw = np.polynomial.legendre.leggauss(max(space.N + 2, 4))
    averages = np.zeros((nsub, space.K), dtype=float)
    for j in range(nsub):
        a = -1.0 + 2.0 * j / nsub
        b = -1.0 + 2.0 * (j + 1) / nsub
        mapped = 0.5 * (b - a) * qx + 0.5 * (a + b)
        basis = np.array(
            [[JacobiP(x, n, 0.0, 0.0) for n in range(space.Nldof)] for x in mapped]
        )
        values = np.dot(basis, u_modal)
        averages[j, :] = 0.5 * np.dot(qw, values)
    return averages


def troubled_cell_mask(sensor_result, threshold=0.0):
    """Extract a boolean troubled-cell mask from a sensor result or array."""
    if hasattr(sensor_result, "activation"):
        activation = np.asarray(sensor_result.activation)
        return activation > threshold
    values = np.asarray(sensor_result)
    return values > threshold


def modal_subcell_fallback(u_modal, troubled, *, keep_modes=0):
    """Optionally flatten high modes in troubled cells.

    The operation is conservative because mode zero is untouched.  It is an
    explicit fallback for experiments that want a subcell-like modal reset;
    the default shock-capture example uses artificial viscosity instead.
    """
    u_modal = np.asarray(u_modal, dtype=float)
    troubled = np.asarray(troubled, dtype=bool)
    if u_modal.ndim != 2 or troubled.shape != (u_modal.shape[1],):
        raise ValueError("dimensões incompatíveis entre u_modal e troubled.")
    keep_modes = int(keep_modes)
    if keep_modes < 0 or keep_modes >= u_modal.shape[0]:
        raise ValueError("keep_modes deve estar entre 0 e N.")

    result = u_modal.copy()
    result[keep_modes + 1 :, troubled] = 0.0
    return result
