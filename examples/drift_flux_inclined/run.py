"""Open-pipe, uphill barotropic drift-flux example using the DG1D toolkit.

Run from the repository root:

    python -m examples.drift_flux_inclined.run

This is the third learning scenario for the reduced two-phase model:
    1) scalar drift-flux kinematic wave;
    2) periodic three-equation barotropic model;
    3) three-equation model with a prescribed inlet, transmissive outlet,
       and gravity in an uphill pipe.

The state U=[m_l,m_g,G] is still barotropic and source-free except for the
mixture gravity term.  Wall friction, energy and regime-dependent closures
remain intentionally absent, so the influence of gravity is easy to see.
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
# 1. Reduced barotropic physics (nondimensional numerical-study parameters)
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

# The pipe rises in the positive-x direction.  Therefore gravity removes
# positive mixture momentum: S_G = -(m_l+m_g) g* sin(theta).
gravity_scaled = 0.45
inclination_deg = 25.0
sin_theta = np.sin(np.deg2rad(inclination_deg))

# Inlet is a liquid-dominant mixture.  A smoother, gas-rich packet is initially
# present farther downstream, allowing us to see its uphill propagation and the
# associated pressure signal before it reaches the outlet.
alpha_inlet = 0.20
j_inlet = 0.45
alpha_packet_amplitude = 0.18
packet_center = 0.60
packet_width = 0.11
final_time = 0.80


# -----------------------------------------------------------------------------
# 2. DG configuration
# -----------------------------------------------------------------------------
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
    """L2 projection of values shaped ``(nip, K)`` into modal coefficients."""
    return space.InvM[:, None] * np.dot(space.wi * space.psi.T, values)


def reconstruct(space: DGSpace1D, U_modal: list[np.ndarray]) -> list[np.ndarray]:
    return [space.psi @ field for field in U_modal]


def project_nonlinear(space: DGSpace1D, values: np.ndarray) -> np.ndarray:
    """L2 projection from the superintegration grid used for source terms."""
    return space.InvM[:, None] * np.dot(space.nlwi * space.nlpsi.T, values)


def inlet_conservative_state() -> np.ndarray:
    """Constant primitive inlet converted into the three conservative variables."""
    return np.array(
        [
            float(np.asarray(value))
            for value in BarotropicDriftFlux.conservative_from_j(
                np.asarray(alpha_inlet),
                np.asarray(rho_g0),
                np.asarray(j_inlet),
                rho_l=rho_l,
                distribution_parameter=C0,
                drift_velocity=Vgj,
            )
        ]
    )


U_INLET = inlet_conservative_state()


def enforce_admissibility(space: DGSpace1D, U_modal: list[np.ndarray]) -> list[np.ndarray]:
    """Keep phase masses in the domain where primitive recovery is valid."""
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


def state_with_open_ghosts(space: DGSpace1D, U_modal: list[np.ndarray]) -> list[np.ndarray]:
    """Build a prescribed-inlet / transmissive-outlet ghost-cell state.

    The inlet ghost is constant and imposes the three supplied conservative
    inlet values.  The right ghost copies the last interior polynomial.  This
    is a deliberately simple weak BC for the subcritical teaching case, not a
    full characteristic boundary treatment.
    """
    padded = []
    for index, field in enumerate(U_modal):
        field_t = np.zeros((space.Nldof, space.K + 2))
        field_t[:, 1:-1] = field
        left_trace = space.Flkp1[0, :] @ field_t[:, 1]
        field_t[0, 0] = 2.0 * U_INLET[index] - left_trace
        field_t[:, -1] = field_t[:, -2]
        padded.append(field_t)
    return padded


def local_rusanov_speeds(space: DGSpace1D, U_t: list[np.ndarray]) -> np.ndarray:
    """Finite-difference spectral-radius bound at all open-pipe interfaces."""
    left_trace = [space.Flkp1[0, :] @ field[:, 1:-1] for field in U_t]
    right_trace = [space.Frk[0, :] @ field[:, 1:-1] for field in U_t]
    lambda_left = BarotropicDriftFlux.spectral_radius(*left_trace, **MODEL_KWARGS)
    lambda_right = BarotropicDriftFlux.spectral_radius(*right_trace, **MODEL_KWARGS)
    lambda_inlet = float(
        BarotropicDriftFlux.spectral_radius(
            *(np.asarray(field[0, 0]) for field in U_t), **MODEL_KWARGS
        )
    )

    C_local = np.empty(space.K + 1)
    C_local[0] = max(lambda_inlet, lambda_left[0])
    C_local[1:-1] = np.maximum(lambda_right[:-1], lambda_left[1:])
    C_local[-1] = lambda_right[-1]
    return 1.05 * C_local


def gravity_source_projection(space: DGSpace1D, U_modal: list[np.ndarray]) -> np.ndarray:
    """Project S_G = -(m_l+m_g) g* sin(theta) with superintegration."""
    m_l = space.nlpsi @ U_modal[0]
    m_g = space.nlpsi @ U_modal[1]
    source_G = -(m_l + m_g) * gravity_scaled * sin_theta
    return project_nonlinear(space, source_G)


def Lh_open_gravity(
    U_modal: list[np.ndarray], t: float, space: DGSpace1D
) -> list[np.ndarray]:
    """DG RHS: conservative fluxes plus the uphill gravity source in G."""
    U_t = state_with_open_ghosts(space, U_modal)
    F_t = FluxProjection(space, U_t, BarotropicDriftFlux.flux, **MODEL_KWARGS)

    # Projected physical-cell fluxes need explicit ghost values.  Left ghost is
    # constant; right ghost is transmissive and copies the last flux polynomial.
    inlet_flux = BarotropicDriftFlux.flux(
        *(np.asarray(field[0, 0]) for field in U_t), **MODEL_KWARGS
    )
    for flux, value in zip(F_t, inlet_flux):
        flux[:, 0] = 0.0
        flux[0, 0] = float(np.asarray(value))
        flux[:, -1] = flux[:, -2]

    C_local = local_rusanov_speeds(space, U_t)
    rhs = []
    for state, flux in zip(U_t, F_t):
        volume = space.S.T @ flux[:, 1:-1]
        interface = rusanov(space, state, flux, C_local)
        rhs.append(
            space.InvM[:, None] * space.J[None, :] ** (-1) * (volume + interface)
        )
    rhs[2] = rhs[2] + gravity_source_projection(space, U_modal)
    return rhs


def global_wave_speed(space: DGSpace1D, U_modal: list[np.ndarray]) -> float:
    fields = reconstruct(space, U_modal)
    return float(np.max(BarotropicDriftFlux.spectral_radius(*fields, **MODEL_KWARGS)))


def field_at(space: DGSpace1D, modal_field: np.ndarray, x_sensor: float) -> float:
    """Sample the nearest available quadrature point to a requested sensor."""
    values = space.psi @ modal_field
    idx = np.unravel_index(np.argmin(np.abs(space.xc - x_sensor)), space.xc.shape)
    return float(values[idx])


def main() -> None:
    space = DGSpace1D(K=K, N=N, xmin=0.0, xmax=L, quad_type="GL")
    xq = space.xc
    alpha_initial = alpha_inlet + alpha_packet_amplitude * np.exp(
        -((xq - packet_center) / packet_width) ** 2
    )
    m_l0, m_g0, G0 = BarotropicDriftFlux.conservative_from_j(
        alpha_initial,
        rho_g0,
        j_inlet,
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

    # The sensor lies on the expected path of the initial gas-rich packet.
    sensor_x = 1.00
    time_history = [0.0]
    alpha_sensor_history = []
    pressure_sensor_history = []
    t = 0.0
    step = 0
    dx = L / K

    print("=" * 80)
    print("Open uphill pipe: barotropic drift-flux U = [m_l, m_g, G]")
    print("Left: prescribed inlet | right: transmissive outlet | gravity in +x uphill")
    print(f"K={K}, N={N}, CFL={CFL}, final time={final_time:.3f}")
    print(f"Inclination={inclination_deg:.1f} deg, g*={gravity_scaled:.3f}")
    print(f"Inlet alpha_g={alpha_inlet:.3f}, inlet j={j_inlet:.3f}")
    print("=" * 80)

    def record_sensor() -> None:
        q_sensor = BarotropicDriftFlux.primitives(
            *(np.asarray(field_at(space, field, sensor_x)) for field in U_modal),
            **MODEL_KWARGS,
        )
        alpha_sensor_history.append(float(np.asarray(q_sensor["alpha_g"])))
        pressure_sensor_history.append(float(np.asarray(q_sensor["pressure"])))

    record_sensor()
    while t < final_time - 1.0e-14:
        lambda_max = max(global_wave_speed(space, U_modal), 1.0e-12)
        dt = min(CFL * dx / ((2 * N + 1) * 1.05 * lambda_max), final_time - t)
        rhs = lambda state, time: Lh_open_gravity(state, time, space)
        U_modal = RKSSP54_Step(U_modal, t, dt, rhs, limiter_func=None)
        U_modal = enforce_admissibility(space, U_modal)
        t += dt
        step += 1

        if step % 10 == 0 or t >= final_time - 1.0e-14:
            time_history.append(t)
            record_sensor()
        if step % 150 == 0 or t >= final_time - 1.0e-14:
            q = BarotropicDriftFlux.primitives(*reconstruct(space, U_modal), **MODEL_KWARGS)
            print(
                f"step {step:4d}: t={t:.4f}, dt={dt:.2e}, "
                f"alpha_g=[{q['alpha_g'].min():.4f}, {q['alpha_g'].max():.4f}], "
                f"j=[{q['j'].min():.4f}, {q['j'].max():.4f}]"
            )

    x_plot = space.xc.T.reshape(-1)
    q0 = BarotropicDriftFlux.primitives(*reconstruct(space, U_initial), **MODEL_KWARGS)
    qf = BarotropicDriftFlux.primitives(*reconstruct(space, U_modal), **MODEL_KWARGS)
    print("-" * 80)
    print(f"Completed {step} adaptive-CFL steps")
    print(f"Final alpha_g range = [{qf['alpha_g'].min():.6f}, {qf['alpha_g'].max():.6f}]")
    print(f"Final mixture velocity range = [{qf['j'].min():.6f}, {qf['j'].max():.6f}]")

    fig, axes = plt.subplots(4, 1, figsize=(10, 11), constrained_layout=True)
    axes[0].plot(x_plot, q0["alpha_g"].T.reshape(-1), "k--", lw=1.5, label="Initial")
    axes[0].plot(x_plot, qf["alpha_g"].T.reshape(-1), color="#0072B2", lw=1.8, label="Final DG")
    axes[0].axvline(sensor_x, color="0.45", ls=":", lw=1.1, label="Sensor")
    axes[0].set_title("Open uphill pipe: gravity-driven barotropic drift-flux")
    axes[0].set_ylabel(r"Gas void fraction $\alpha_g$")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_xlim(0.0, L)
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="best", ncol=3)

    axes[1].plot(x_plot, q0["pressure"].T.reshape(-1), "k--", lw=1.5, label="Initial")
    axes[1].plot(x_plot, qf["pressure"].T.reshape(-1), color="#D55E00", lw=1.8, label="Final DG")
    axes[1].set_ylabel("Barotropic pressure")
    axes[1].set_xlim(0.0, L)
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="best")

    axes[2].plot(x_plot, q0["j"].T.reshape(-1), "k--", lw=1.4, label=r"Initial $j$")
    axes[2].plot(x_plot, qf["j"].T.reshape(-1), color="#E69F00", lw=1.8, label=r"Final $j$")
    axes[2].plot(x_plot, qf["u_l"].T.reshape(-1), color="#009E73", lw=1.4, label=r"Final $u_l$")
    axes[2].plot(x_plot, qf["u_g"].T.reshape(-1), color="#CC79A7", lw=1.4, label=r"Final $u_g$")
    axes[2].set_ylabel("Velocity")
    axes[2].set_xlim(0.0, L)
    axes[2].grid(alpha=0.3)
    axes[2].legend(loc="best", ncol=4)

    axes[3].plot(time_history, alpha_sensor_history, color="#0072B2", lw=1.8, label=r"$\alpha_g$ at sensor")
    axes[3].set_xlabel("Time")
    axes[3].set_ylabel(r"Gas void fraction at $x_s$")
    axes[3].grid(alpha=0.3)
    pressure_axis = axes[3].twinx()
    pressure_axis.plot(time_history, pressure_sensor_history, color="#D55E00", lw=1.6, label="Pressure at sensor")
    pressure_axis.set_ylabel("Barotropic pressure at $x_s$")
    lines, labels = axes[3].get_legend_handles_labels()
    lines2, labels2 = pressure_axis.get_legend_handles_labels()
    axes[3].legend(lines + lines2, labels + labels2, loc="best")

    for axis in axes[:3]:
        axis.set_xlabel("x")

    fig.savefig(OUTPUT_DIR / "driftflux_barotropic_gravity_result.png", dpi=180, bbox_inches="tight")
    np.savez(
        OUTPUT_DIR / "driftflux_barotropic_gravity_results.npz",
        x=x_plot,
        alpha_initial=q0["alpha_g"].T.reshape(-1),
        alpha_final=qf["alpha_g"].T.reshape(-1),
        pressure_initial=q0["pressure"].T.reshape(-1),
        pressure_final=qf["pressure"].T.reshape(-1),
        liquid_velocity_final=qf["u_l"].T.reshape(-1),
        gas_velocity_final=qf["u_g"].T.reshape(-1),
        mixture_velocity_initial=q0["j"].T.reshape(-1),
        mixture_velocity_final=qf["j"].T.reshape(-1),
        time=np.asarray(time_history),
        alpha_sensor=np.asarray(alpha_sensor_history),
        pressure_sensor=np.asarray(pressure_sensor_history),
        sensor_x=sensor_x,
        inclination_deg=inclination_deg,
        gravity_scaled=gravity_scaled,
    )
    plt.show()


if __name__ == "__main__":
    main()
