import numpy as np

from dg1d_toolkit import DGSpace1D
from dg1d_toolkit.barotropic_driftflux import BarotropicDriftFlux
from dg1d_toolkit.twophase_fluxes import DriftFlux


def test_modal_space_has_consistent_shapes():
    space = DGSpace1D(K=4, N=2, xmin=0.0, xmax=1.0, quad_type="GL")
    assert space.Nldof == 3
    assert space.psi.shape == (space.nip, space.Nldof)
    assert space.xc.shape == (space.nip, space.K)
    assert np.all(space.J > 0.0)


def test_scalar_drift_flux_is_finite():
    alpha = np.array([0.1, 0.4, 0.8])
    flux = DriftFlux.gas_void_fraction_flux(alpha, 0.8, 1.2, 0.35)
    speed = DriftFlux.gas_void_fraction_speed(alpha, 0.8, 1.2, 0.35)
    assert np.all(np.isfinite(flux))
    assert np.all(np.isfinite(speed))


def test_barotropic_primitive_recovery_is_admissible():
    m_l, m_g, momentum = BarotropicDriftFlux.conservative_from_j(
        alpha_g=np.asarray([0.2, 0.4]),
        rho_g=np.asarray([0.2, 0.25]),
        j=np.asarray([0.3, 0.35]),
        rho_l=1.0,
    )
    primitives = BarotropicDriftFlux.primitives(
        m_l, m_g, momentum, rho_l=1.0, sound_speed_g=1.2
    )
    assert np.all((primitives["alpha_g"] > 0.0) & (primitives["alpha_g"] < 1.0))
    assert np.all(primitives["pressure"] > 0.0)
