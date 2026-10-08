"""Open uphill barotropic drift-flux model with characteristic boundaries.

Run from the repository root:

    python -m examples.drift_flux_characteristic.run

The model solves U=[m_l,m_g,G] with gravity in an uphill pipe.  It improves
the preceding open-pipe example by using the eigenstructure of dF/dU at each
boundary: only characteristics entering the domain are prescribed externally;
outgoing characteristics are retained from the interior solution.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from dg1d_toolkit.barotropic_driftflux_characteristic import BarotropicDriftFlux
from dg1d_toolkit.dg1d_core import DGSpace1D
from dg1d_toolkit.dg1d_integrators import RKSSP54_Step
from dg1d_toolkit.numericalfluxes import rusanov
from dg1d_toolkit.operators import FluxProjection


OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "generated"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# 1. Reduced physics and open-pipe setup (nondimensional parameters)
# -----------------------------------------------------------------------------
L = 2.0
rho_l = 1.0
rho_g0 = 0.20
sound_speed_g = 1.20
C0 = 1.05
Vgj = 0.08
alpha_floor = 1.0e-5
rho_g_floor = 1.0e-5
rho_g_ceiling = 2.0

gravity_scaled = 0.45
inclination_deg = 25.0
sin_theta = np.sin(np.deg2rad(inclination_deg))

alpha_inlet = 0.20
j_inlet = 0.45
outlet_pressure = sound_speed_g**2 * rho_g0

alpha_packet_amplitude = 0.18
packet_center = 0.60
packet_width = 0.11
sensor_x = 1.00
final_time = 0.80

K = 140
N = 1
CFL = 0.18

MODEL_KWARGS = dict(
    rho_l=rho_l,
    sound_speed_g=sound_speed_g,
    distribution_parameter=C0,
    drift_velocity=Vgj,
    alpha_floor=alpha_floor,
    rho_g_floor=rho_g_floor,
)


def project_to_modal(space: DGSpace1D, values: np.ndarray) -> np.ndarray:
    return space.InvM[:, None] * np.dot(space.wi * space.psi.T, values)


def reconstruct(space: DGSpace1D, U_modal: list[np.ndarray]) -> list[np.ndarray]:
    return [space.psi @ field for field in U_modal]


def project_nonlinear(space: DGSpace1D, values: np.ndarray) -> np.ndarray:
    return space.InvM[:, None] * np.dot(space.nlwi * space.nlpsi.T, values)


def conservative_from_primitives(alpha_g: float, rho_g: float, j: float) -> np.ndarray:
    """Convenience wrapper returning a float state [m_l, m_g, G]."""
    return np.asarray(
        BarotropicDriftFlux.conservative_from_j(
            np.asarray(alpha_g),
            np.asarray(rho_g),
            np.asarray(j),
            rho_l=rho_l,
            distribution_parameter=C0,
            drift_velocity=Vgj,
        ),
        dtype=float,
    )


U_INLET = conservative_from_primitives(alpha_inlet, rho_g0, j_inlet)


def admissible_state(U: np.ndarray) -> np.ndarray:
    """Return a physically admissible constant state for a ghost cell."""
    state = np.asarray(U, dtype=float).copy()
    state[0] = np.clip(state[0], alpha_floor * rho_l, (1.0 - alpha_floor) * rho_l)
    alpha_g = 1.0 - state[0] / rho_l
    state[1] = np.clip(state[1], rho_g_floor * alpha_g, rho_g_ceiling * alpha_g)
    return state


def characteristic_ghost(
    U_inside: np.ndarray, U_target: np.ndarray, boundary: str
) -> np.ndarray:
    """Apply target data only to characteristics entering the domain.

    At x=0, positive eigenvalues travel into the domain and are prescribed
    from ``U_target``.  At x=L, negative eigenvalues enter from the exterior;
    only those receive the target contribution.  All other characteristic
    components are copied from ``U_inside`` and therefore leave freely.
    """
    A = BarotropicDriftFlux.flux_jacobian(*U_inside, **MODEL_KWARGS)
    eigenvalues, right_eigenvectors = np.linalg.eig(A)
    eigenvalues = np.real_if_close(eigenvalues, tol=1000).real
    right_eigenvectors = np.real_if_close(right_eigenvectors, tol=1000).real

    characteristic_jump = np.linalg.solve(right_eigenvectors, U_target - U_inside)
    if boundary == "left":
        incoming = eigenvalues > 1.0e-10
    elif boundary == "right":
        incoming = eigenvalues < -1.0e-10
    else:
        raise ValueError("boundary must be 'left' or 'right'.")

    U_boundary = U_inside + right_eigenvectors @ (incoming * characteristic_jump)
    return admissible_state(U_boundary)


def outlet_target(U_inside: np.ndarray) -> np.ndarray:
    """State with exterior pressure but the interior alpha_g and j.

    Its difference from the interior state is projected only onto the incoming
    characteristic at x=L.  It therefore acts as a pressure outlet without
    over-constraining the other two characteristic waves.
    """
    q = BarotropicDriftFlux.primitives(*U_inside, **MODEL_KWARGS)
    rho_g_out = outlet_pressure / sound_speed_g**2
    return conservative_from_primitives(
        float(np.asarray(q["alpha_g"])),
        rho_g_out,
        float(np.asarray(q["j"])),
    )


def enforce_admissibility(space: DGSpace1D, U_modal: list[np.ndarray]) -> list[np.ndarray]:
    """Bound phase masses before the next primitive recovery."""
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


def characteristic_state_with_ghosts(
    space: DGSpace1D, U_modal: list[np.ndarray]
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Construct constant ghost cells from the local characteristic relations."""
    padded = []
    for field in U_modal:
        field_t = np.zeros((space.Nldof, space.K + 2))
        field_t[:, 1:-1] = field
        padded.append(field_t)

    U_left_inside = np.asarray([space.Flkp1[0, :] @ field[:, 1] for field in padded])
    U_right_inside = np.asarray([space.Frk[0, :] @ field[:, -2] for field in padded])
    U_left_ghost = characteristic_ghost(U_left_inside, U_INLET, boundary="left")
    U_right_ghost = characteristic_ghost(
        U_right_inside, outlet_target(U_right_inside), boundary="right"
    )

    for index, field in enumerate(padded):
        field[:, 0] = 0.0
        field[0, 0] = U_left_ghost[index]
        field[:, -1] = 0.0
        field[0, -1] = U_right_ghost[index]
    return padded, U_left_ghost, U_right_ghost


def local_rusanov_speeds(
    space: DGSpace1D, U_t: list[np.ndarray], U_left_ghost: np.ndarray, U_right_ghost: np.ndarray
) -> np.ndarray:
    """Compute local interface speeds from the numerical flux Jacobian."""
    left_trace = [space.Flkp1[0, :] @ field[:, 1:-1] for field in U_t]
    right_trace = [space.Frk[0, :] @ field[:, 1:-1] for field in U_t]
    lambda_left = BarotropicDriftFlux.spectral_radius(*left_trace, **MODEL_KWARGS)
    lambda_right = BarotropicDriftFlux.spectral_radius(*right_trace, **MODEL_KWARGS)
    lambda_ghost_left = float(BarotropicDriftFlux.spectral_radius(*U_left_ghost, **MODEL_KWARGS))
    lambda_ghost_right = float(BarotropicDriftFlux.spectral_radius(*U_right_ghost, **MODEL_KWARGS))

    C_local = np.empty(space.K + 1)
    C_local[0] = max(lambda_ghost_left, lambda_left[0])
    C_local[1:-1] = np.maximum(lambda_right[:-1], lambda_left[1:])
    C_local[-1] = max(lambda_right[-1], lambda_ghost_right)
    return 1.05 * C_local


def gravity_source_projection(space: DGSpace1D, U_modal: list[np.ndarray]) -> np.ndarray:
    m_l = space.nlpsi @ U_modal[0]
    m_g = space.nlpsi @ U_modal[1]
    return project_nonlinear(space, -(m_l + m_g) * gravity_scaled * sin_theta)


def Lh_characteristic_gravity(
    U_modal: list[np.ndarray], t: float, space: DGSpace1D
) -> list[np.ndarray]:
    """DG operator using characteristic ghosts and the uphill gravity source."""
    U_t, U_left_ghost, U_right_ghost = characteristic_state_with_ghosts(space, U_modal)
    F_t = FluxProjection(space, U_t, BarotropicDriftFlux.flux, **MODEL_KWARGS)
    F_left = BarotropicDriftFlux.flux(*U_left_ghost, **MODEL_KWARGS)
    F_right = BarotropicDriftFlux.flux(*U_right_ghost, **MODEL_KWARGS)
    for flux, f_left, f_right in zip(F_t, F_left, F_right):
        flux[:, 0] = 0.0
        flux[0, 0] = float(np.asarray(f_left))
        flux[:, -1] = 0.0
        flux[0, -1] = float(np.asarray(f_right))

    C_local = local_rusanov_speeds(space, U_t, U_left_ghost, U_right_ghost)
    rhs = []
    for state, flux in zip(U_t, F_t):
        volume = space.S.T @ flux[:, 1:-1]
        face = rusanov(space, state, flux, C_local)
        rhs.append(space.InvM[:, None] * space.J[None, :] ** (-1) * (volume + face))
    rhs[2] = rhs[2] + gravity_source_projection(space, U_modal)
    return rhs


def global_wave_speed(space: DGSpace1D, U_modal: list[np.ndarray]) -> float:
    return float(np.max(BarotropicDriftFlux.spectral_radius(*reconstruct(space, U_modal), **MODEL_KWARGS)))


def sample_primitives(space: DGSpace1D, U_modal: list[np.ndarray], x_sensor: float) -> dict[str, np.ndarray]:
    fields = reconstruct(space, U_modal)
    idx = np.unravel_index(np.argmin(np.abs(space.xc - x_sensor)), space.xc.shape)
    return BarotropicDriftFlux.primitives(*(np.asarray(field[idx]) for field in fields), **MODEL_KWARGS)


def main() -> None:
    space = DGSpace1D(K=K, N=N, xmin=0.0, xmax=L, quad_type="GL")
    alpha_initial = alpha_inlet + alpha_packet_amplitude * np.exp(
        -((space.xc - packet_center) / packet_width) ** 2
    )
    initial_state = BarotropicDriftFlux.conservative_from_j(
        alpha_initial, rho_g0, j_inlet,
        rho_l=rho_l, distribution_parameter=C0, drift_velocity=Vgj,
    )
    U_modal = [project_to_modal(space, field) for field in initial_state]
    U_modal = enforce_admissibility(space, U_modal)
    U_initial = [field.copy() for field in U_modal]

    time_history = [0.0]
    alpha_sensor = []
    pressure_sensor = []
    left_pressure = []
    right_pressure = []
    t = 0.0
    step = 0
    dx = L / K

    def record() -> None:
        q_sensor = sample_primitives(space, U_modal, sensor_x)
        q_left = sample_primitives(space, U_modal, 0.0)
        q_right = sample_primitives(space, U_modal, L)
        alpha_sensor.append(float(np.asarray(q_sensor["alpha_g"])))
        pressure_sensor.append(float(np.asarray(q_sensor["pressure"])))
        left_pressure.append(float(np.asarray(q_left["pressure"])))
        right_pressure.append(float(np.asarray(q_right["pressure"])))

    print("=" * 82)
    print("Open uphill pipe with characteristic boundary conditions")
    print("Left: prescribed incoming characteristics | right: pressure outlet")
    print(f"K={K}, N={N}, theta={inclination_deg:.1f} deg, p_out={outlet_pressure:.4f}")
    print("=" * 82)

    record()
    while t < final_time - 1.0e-14:
        lambda_max = max(global_wave_speed(space, U_modal), 1.0e-12)
        dt = min(CFL * dx / ((2 * N + 1) * 1.05 * lambda_max), final_time - t)
        rhs = lambda state, time: Lh_characteristic_gravity(state, time, space)
        U_modal = RKSSP54_Step(U_modal, t, dt, rhs, limiter_func=None)
        U_modal = enforce_admissibility(space, U_modal)
        t += dt
        step += 1

        if step % 10 == 0 or t >= final_time - 1.0e-14:
            time_history.append(t)
            record()
        if step % 150 == 0 or t >= final_time - 1.0e-14:
            q = BarotropicDriftFlux.primitives(*reconstruct(space, U_modal), **MODEL_KWARGS)
            print(
                f"step {step:4d}: t={t:.4f}, dt={dt:.2e}, "
                f"alpha_g=[{q['alpha_g'].min():.4f}, {q['alpha_g'].max():.4f}], "
                f"p=[{q['pressure'].min():.4f}, {q['pressure'].max():.4f}]"
            )

    x_plot = space.xc.T.reshape(-1)
    q0 = BarotropicDriftFlux.primitives(*reconstruct(space, U_initial), **MODEL_KWARGS)
    qf = BarotropicDriftFlux.primitives(*reconstruct(space, U_modal), **MODEL_KWARGS)
    print("-" * 82)
    print(f"Completed {step} adaptive-CFL steps")
    print(f"Final alpha_g range = [{qf['alpha_g'].min():.6f}, {qf['alpha_g'].max():.6f}]")
    print(f"Final p range = [{qf['pressure'].min():.6f}, {qf['pressure'].max():.6f}]")

    fig, axes = plt.subplots(4, 1, figsize=(10, 11), constrained_layout=True)
    axes[0].plot(x_plot, q0["alpha_g"].T.reshape(-1), "k--", lw=1.5, label="Initial")
    axes[0].plot(x_plot, qf["alpha_g"].T.reshape(-1), color="#0072B2", lw=1.8, label="Final DG")
    axes[0].axvline(sensor_x, color="0.45", ls=":", lw=1.1, label="Sensor")
    axes[0].set_title("Open uphill pipe with characteristic boundary conditions")
    axes[0].set_ylabel(r"Gas void fraction $\alpha_g$")
    axes[0].set_xlim(0.0, L)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="best", ncol=3)

    axes[1].plot(x_plot, q0["pressure"].T.reshape(-1), "k--", lw=1.5, label="Initial")
    axes[1].plot(x_plot, qf["pressure"].T.reshape(-1), color="#D55E00", lw=1.8, label="Final DG")
    axes[1].axhline(outlet_pressure, color="0.35", ls=":", lw=1.2, label=r"Target $p_\mathrm{out}$")
    axes[1].set_ylabel("Barotropic pressure")
    axes[1].set_xlim(0.0, L)
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="best")

    axes[2].plot(x_plot, qf["j"].T.reshape(-1), color="#E69F00", lw=1.8, label=r"$j$")
    axes[2].plot(x_plot, qf["u_l"].T.reshape(-1), color="#009E73", lw=1.5, label=r"$u_l$")
    axes[2].plot(x_plot, qf["u_g"].T.reshape(-1), color="#CC79A7", lw=1.5, label=r"$u_g$")
    axes[2].set_ylabel("Final velocity")
    axes[2].set_xlim(0.0, L)
    axes[2].grid(alpha=0.3)
    axes[2].legend(loc="best", ncol=3)

    axes[3].plot(time_history, alpha_sensor, color="#0072B2", lw=1.8, label=r"$\alpha_g$ at sensor")
    axes[3].set_xlabel("Time")
    axes[3].set_ylabel(r"Gas void fraction at $x_s$")
    axes[3].grid(alpha=0.3)
    right_axis = axes[3].twinx()
    right_axis.plot(time_history, pressure_sensor, color="#D55E00", lw=1.6, label="Pressure at sensor")
    right_axis.plot(time_history, left_pressure, color="#009E73", lw=1.2, ls="--", label="Pressure near inlet")
    right_axis.plot(time_history, right_pressure, color="#CC79A7", lw=1.2, ls="--", label="Pressure near outlet")
    right_axis.set_ylabel("Barotropic pressure")
    handles, labels = axes[3].get_legend_handles_labels()
    handles2, labels2 = right_axis.get_legend_handles_labels()
    axes[3].legend(handles + handles2, labels + labels2, loc="best", ncol=2)

    for axis in axes[:3]:
        axis.set_xlabel("x")

    fig.savefig(
        OUTPUT_DIR / "driftflux_barotropic_gravity_characteristic_result.png",
        dpi=180,
        bbox_inches="tight",
    )
    np.savez(
        OUTPUT_DIR / "driftflux_barotropic_gravity_characteristic_results.npz",
        x=x_plot,
        alpha_initial=q0["alpha_g"].T.reshape(-1),
        alpha_final=qf["alpha_g"].T.reshape(-1),
        pressure_initial=q0["pressure"].T.reshape(-1),
        pressure_final=qf["pressure"].T.reshape(-1),
        liquid_velocity_final=qf["u_l"].T.reshape(-1),
        gas_velocity_final=qf["u_g"].T.reshape(-1),
        mixture_velocity_final=qf["j"].T.reshape(-1),
        time=np.asarray(time_history),
        alpha_sensor=np.asarray(alpha_sensor),
        pressure_sensor=np.asarray(pressure_sensor),
        pressure_inlet=np.asarray(left_pressure),
        pressure_outlet=np.asarray(right_pressure),
        outlet_pressure_target=outlet_pressure,
        sensor_x=sensor_x,
    )
    plt.show()


if __name__ == "__main__":
    main()
