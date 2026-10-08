"""DG simulation of a minimal three-equation barotropic drift-flux model.

Run from the repository root directory:

    python -m examples.drift_flux_barotropic.run

The conservative state is U = [m_l, m_g, G], with two phase mass balances and
one mixture momentum balance.  The example is periodic and source-free so
that conservation and primitive recovery can be checked before adding gravity,
friction, energy, or realistic flow-regime closures.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from dg1d_toolkit.barotropic_driftflux import BarotropicDriftFlux
from dg1d_toolkit.dg1d_core import DGSpace1D
from dg1d_toolkit.dg1d_integrators import RKSSP54_Step
from dg1d_toolkit.numericalfluxes import rusanov
from dg1d_toolkit.operators import FluxProjection


OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "generated"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# 1. Reduced barotropic drift-flux physics (nondimensional study parameters)
# -----------------------------------------------------------------------------
L = 1.0
rho_l = 1.0
rho_g0 = 0.20
sound_speed_g = 1.20
C0 = 1.05
Vgj = 0.08
alpha_floor = 1.0e-5
rho_g_floor = 1.0e-5
rho_g_ceiling = 2.0

# Smooth gas-rich packet in a periodic pipe.  Initial pressure is uniform
# because rho_g is uniform.  Subsequent pressure/velocity changes are created
# by the coupled mass and mixture-momentum equations.
alpha_base = 0.20
alpha_amplitude = 0.18
packet_center = 0.30
packet_width = 0.075
j0 = 0.35
final_time = 0.35


# -----------------------------------------------------------------------------
# 2. DG configuration
# -----------------------------------------------------------------------------
K = 120
N = 1
CFL = 0.20

MODEL_KWARGS = dict(
    rho_l=rho_l,
    sound_speed_g=sound_speed_g,
    distribution_parameter=C0,
    drift_velocity=Vgj,
    alpha_floor=alpha_floor,
    rho_g_floor=rho_g_floor,
)


def project_to_modal(space: DGSpace1D, values: np.ndarray) -> np.ndarray:
    """L2-project quadrature values with shape (nip, K) into modal DOFs."""
    return space.InvM[:, None] * np.dot(space.wi * space.psi.T, values)


def reconstruct(space: DGSpace1D, U_modal: list[np.ndarray]) -> list[np.ndarray]:
    """Reconstruct all conservative fields at the ordinary quadrature points."""
    return [space.psi @ field for field in U_modal]


def enforce_admissibility(space: DGSpace1D, U_modal: list[np.ndarray]) -> list[np.ndarray]:
    """Bound phase masses before primitive recovery.

    The smooth benchmark should not activate this correction.  It is a clear
    numerical guard for later experiments with sharper initial conditions.
    A production solver should replace it with a conservative positivity
    limiter that acts during every RK stage.
    """
    m_l, m_g, G = reconstruct(space, U_modal)
    m_l_bounded = np.clip(m_l, alpha_floor * rho_l, (1.0 - alpha_floor) * rho_l)
    alpha_g = 1.0 - m_l_bounded / rho_l
    m_g_bounded = np.clip(m_g, rho_g_floor * alpha_g, rho_g_ceiling * alpha_g)

    if np.array_equal(m_l, m_l_bounded) and np.array_equal(m_g, m_g_bounded):
        return U_modal
    return [
        project_to_modal(space, m_l_bounded),
        project_to_modal(space, m_g_bounded),
        project_to_modal(space, G),
    ]


def periodic_state_with_ghosts(
    space: DGSpace1D, U_modal: list[np.ndarray]
) -> list[np.ndarray]:
    """Pad every modal field with periodic ghost cells."""
    padded = []
    for field in U_modal:
        field_t = np.zeros((space.Nldof, space.K + 2))
        field_t[:, 1:-1] = field
        field_t[:, 0] = field[:, -1]
        field_t[:, -1] = field[:, 0]
        padded.append(field_t)
    return padded


def local_rusanov_speeds(space: DGSpace1D, U_t: list[np.ndarray]) -> np.ndarray:
    """Compute a local spectral-radius bound at all periodic interfaces."""
    left_trace = [space.Flkp1[0, :] @ field[:, 1:-1] for field in U_t]
    right_trace = [space.Frk[0, :] @ field[:, 1:-1] for field in U_t]
    lambda_left = BarotropicDriftFlux.spectral_radius(*left_trace, **MODEL_KWARGS)
    lambda_right = BarotropicDriftFlux.spectral_radius(*right_trace, **MODEL_KWARGS)

    C_local = np.empty(space.K + 1)
    periodic_speed = max(lambda_right[-1], lambda_left[0])
    C_local[0] = periodic_speed
    C_local[1:-1] = np.maximum(lambda_right[:-1], lambda_left[1:])
    C_local[-1] = periodic_speed
    return 1.05 * C_local


def Lh_barotropic_driftflux(
    U_modal: list[np.ndarray], t: float, space: DGSpace1D
) -> list[np.ndarray]:
    """DG operator for the source-free periodic three-equation system."""
    U_t = periodic_state_with_ghosts(space, U_modal)
    F_t = FluxProjection(space, U_t, BarotropicDriftFlux.flux, **MODEL_KWARGS)

    # The periodic ghost flux uses the modal flux of the adjacent physical cell.
    for flux in F_t:
        flux[:, 0] = flux[:, -2]
        flux[:, -1] = flux[:, 1]

    C_local = local_rusanov_speeds(space, U_t)
    rhs = []
    for state, flux in zip(U_t, F_t):
        volume = space.S.T @ flux[:, 1:-1]
        interface = rusanov(space, state, flux, C_local)
        rhs.append(
            space.InvM[:, None] * space.J[None, :] ** (-1) * (volume + interface)
        )
    return rhs


def global_wave_speed(space: DGSpace1D, U_modal: list[np.ndarray]) -> float:
    """Maximum numerical characteristic speed for the adaptive CFL step."""
    fields = reconstruct(space, U_modal)
    return float(np.max(BarotropicDriftFlux.spectral_radius(*fields, **MODEL_KWARGS)))


def integral(space: DGSpace1D, modal_field: np.ndarray) -> float:
    """Domain integral of a modal field; mode zero is the cell average."""
    return float(np.sum(2.0 * space.J * modal_field[0, :]))


def main() -> None:
    space = DGSpace1D(K=K, N=N, xmin=0.0, xmax=L, quad_type="GL")
    xq = space.xc
    alpha_initial = alpha_base + alpha_amplitude * np.exp(
        -((xq - packet_center) / packet_width) ** 2
    )
    m_l0, m_g0, G0 = BarotropicDriftFlux.conservative_from_j(
        alpha_initial,
        rho_g0,
        j0,
        rho_l=rho_l,
        distribution_parameter=C0,
        drift_velocity=Vgj,
    )
    U_modal = [
        project_to_modal(space, m_l0),
        project_to_modal(space, m_g0),
        project_to_modal(space, G0),
    ]
    U_modal = enforce_admissibility(space, U_modal)
    U_initial = [field.copy() for field in U_modal]

    conserved_initial = np.array([integral(space, field) for field in U_modal])
    time_history = [0.0]
    residual_history = [np.zeros(3)]
    t = 0.0
    step = 0
    dx = L / K

    print("=" * 76)
    print("DG barotropic drift-flux test: U = [m_l, m_g, G]")
    print("Periodic, source-free: two phase masses and mixture momentum")
    print(f"K={K}, N={N}, CFL={CFL}, final time={final_time:.3f}")
    print(f"rho_l={rho_l:.3f}, rho_g0={rho_g0:.3f}, c_g={sound_speed_g:.3f}")
    print(f"C0={C0:.3f}, Vgj={Vgj:.3f}, j0={j0:.3f}")
    print("=" * 76)

    while t < final_time - 1.0e-14:
        lambda_max = max(global_wave_speed(space, U_modal), 1.0e-12)
        dt = min(CFL * dx / ((2 * N + 1) * 1.05 * lambda_max), final_time - t)
        rhs = lambda state, time: Lh_barotropic_driftflux(state, time, space)
        U_modal = RKSSP54_Step(U_modal, t, dt, rhs, limiter_func=None)
        U_modal = enforce_admissibility(space, U_modal)
        t += dt
        step += 1

        if step % 10 == 0 or t >= final_time - 1.0e-14:
            conserved = np.array([integral(space, field) for field in U_modal])
            time_history.append(t)
            residual_history.append(conserved - conserved_initial)
        if step % 100 == 0 or t >= final_time - 1.0e-14:
            q = BarotropicDriftFlux.primitives(*reconstruct(space, U_modal), **MODEL_KWARGS)
            print(
                f"step {step:4d}: t={t:.4f}, dt={dt:.2e}, "
                f"alpha_g=[{q['alpha_g'].min():.4f}, {q['alpha_g'].max():.4f}], "
                f"p=[{q['pressure'].min():.4f}, {q['pressure'].max():.4f}]"
            )

    x_plot = space.xc.T.reshape(-1)
    q0 = BarotropicDriftFlux.primitives(*reconstruct(space, U_initial), **MODEL_KWARGS)
    qf = BarotropicDriftFlux.primitives(*reconstruct(space, U_modal), **MODEL_KWARGS)
    final_residual = np.array([integral(space, field) for field in U_modal]) - conserved_initial

    print("-" * 76)
    print(f"Completed {step} adaptive-CFL steps")
    print("Conservation residual [m_l, m_g, G] = " + np.array2string(final_residual, precision=3))
    print("alpha_g final range = " f"[{qf['alpha_g'].min():.6f}, {qf['alpha_g'].max():.6f}]")

    fig, axes = plt.subplots(4, 1, figsize=(10, 11), constrained_layout=True)
    axes[0].plot(x_plot, q0["alpha_g"].T.reshape(-1), "k--", lw=1.5, label="Initial")
    axes[0].plot(x_plot, qf["alpha_g"].T.reshape(-1), color="#0072B2", lw=1.8, label="Final DG")
    axes[0].set_ylabel(r"Gas void fraction $\alpha_g$")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_xlim(0.0, L)
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="best")
    axes[0].set_title("Three-equation barotropic drift-flux: smooth gas-rich packet")

    axes[1].plot(x_plot, q0["pressure"].T.reshape(-1), "k--", lw=1.5, label="Initial")
    axes[1].plot(x_plot, qf["pressure"].T.reshape(-1), color="#D55E00", lw=1.8, label="Final DG")
    axes[1].set_ylabel("Barotropic pressure")
    axes[1].set_xlim(0.0, L)
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="best")

    axes[2].plot(x_plot, qf["u_l"].T.reshape(-1), color="#009E73", lw=1.7, label=r"$u_l$")
    axes[2].plot(x_plot, qf["u_g"].T.reshape(-1), color="#CC79A7", lw=1.7, label=r"$u_g$")
    axes[2].plot(x_plot, qf["j"].T.reshape(-1), color="#E69F00", lw=1.4, ls="--", label=r"$j$")
    axes[2].set_ylabel("Velocity")
    axes[2].set_xlim(0.0, L)
    axes[2].grid(alpha=0.3)
    axes[2].legend(loc="best", ncol=3)

    residual_array = np.asarray(residual_history)
    axes[3].plot(time_history, residual_array[:, 0], lw=1.5, label=r"$\Delta\int m_l dx$")
    axes[3].plot(time_history, residual_array[:, 1], lw=1.5, label=r"$\Delta\int m_g dx$")
    axes[3].plot(time_history, residual_array[:, 2], lw=1.5, label=r"$\Delta\int G dx$")
    axes[3].axhline(0.0, color="k", lw=0.7)
    axes[3].set_xlabel("Time")
    axes[3].set_ylabel("Periodic conservation residual")
    axes[3].grid(alpha=0.3)
    axes[3].legend(loc="best", ncol=3)

    for axis in axes[:3]:
        axis.set_xlabel("x [m]")

    fig.savefig(OUTPUT_DIR / "driftflux_barotropic_result.png", dpi=180, bbox_inches="tight")
    np.savez(
        OUTPUT_DIR / "driftflux_barotropic_results.npz",
        x=x_plot,
        alpha_initial=q0["alpha_g"].T.reshape(-1),
        alpha_final=qf["alpha_g"].T.reshape(-1),
        pressure_initial=q0["pressure"].T.reshape(-1),
        pressure_final=qf["pressure"].T.reshape(-1),
        liquid_velocity_final=qf["u_l"].T.reshape(-1),
        gas_velocity_final=qf["u_g"].T.reshape(-1),
        mixture_velocity_final=qf["j"].T.reshape(-1),
        time=np.asarray(time_history),
        conservation_residual=np.asarray(residual_history),
        final_time=final_time,
        K=K,
        N=N,
        rho_l=rho_l,
        rho_g0=rho_g0,
        sound_speed_g=sound_speed_g,
        C0=C0,
        drift_velocity=Vgj,
    )
    plt.show()


if __name__ == "__main__":
    main()
