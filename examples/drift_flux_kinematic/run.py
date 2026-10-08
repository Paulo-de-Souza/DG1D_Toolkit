"""DG simulation of a reduced 1-D drift-flux kinematic wave.

Run from the repository root with

    python -m examples.drift_flux_kinematic.run

The script is intentionally a first step between the scalar DG examples and
the full two-fluid equations in the ALFASim-style formulation.  It solves

    alpha_g,t + d/dx { alpha_g [ C0 j + Vgj (1-alpha_g) ] } = 0,

where alpha_g is the gas void fraction.  The prescribed mixture superficial
velocity j, distribution parameter C0, and drift velocity Vgj retain a basic
gas-liquid slip mechanism.  Pressure/momentum/energy balances, friction,
gravity and changing thermodynamic properties are intentionally frozen.

The test is a Riemann problem whose exact entropy solution is a single shock.
It exercises the complete hyperbolic path of the toolkit:

    DGSpace1D -> FluxProjection -> local Rusanov -> RKSSP54 -> limiter.

Outputs
-------
driftflux_kinematic_result.png
    Numerical and exact void fraction, liquid holdup, and conservation plot.
driftflux_kinematic_results.npz
    Reusable arrays for later comparisons/animations.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from dg1d_toolkit.dg1d_core import DGSpace1D
from dg1d_toolkit.dg1d_integrators import RKSSP54_Step
from dg1d_toolkit.dg1d_limiters import SlopeLimiterN
from dg1d_toolkit.numericalfluxes import rusanov
from dg1d_toolkit.operators import FluxProjection
from dg1d_toolkit.twophase_fluxes import DriftFlux


OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "generated"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# 1. Physical model: scalar, reduced drift-flux closure
# -----------------------------------------------------------------------------
L = 1.0                              # Pipe length [m]
j_m = 0.80                           # Mixture superficial velocity j [m/s]
C0 = 1.20                            # Distribution parameter [-]
Vgj = 0.35                           # Drift velocity [m/s]

# Riemann data.  For Vgj > 0, the flux is concave.  alpha_L < alpha_R gives
# an entropy shock because f'(alpha_L) > f'(alpha_R).
alpha_left = 0.08                    # Inlet/left void fraction [-]
alpha_right = 0.62                   # Initial right void fraction [-]
x_jump = 0.40                        # Initial discontinuity position [m]
final_time = 0.28                    # Shock remains inside the domain [s]


# -----------------------------------------------------------------------------
# 2. DG / time-integration configuration
# -----------------------------------------------------------------------------
K = 160                              # Elements
N = 2                                # Polynomial degree per element
CFL = 0.35                           # Conservative value for explicit RKDG
alpha_floor = 1.0e-8                 # Physical lower/upper buffer


def project_to_modal(space: DGSpace1D, values: np.ndarray) -> np.ndarray:
    """L2-project nodal/quadrature values of shape (nip, K) into DG modes."""
    rhs = np.dot(space.wi * space.psi.T, values)
    return space.InvM[:, None] * rhs


class VoidFractionLimiter:
    """Toolkit slope limiter plus a bound-preserving projection for alpha_g.

    The generic scalar slope limiter controls the discontinuity.  The extra
    clip/project stage enforces the physically mandatory 0 <= alpha_g <= 1
    invariant at the quadrature points.  This is intentionally visible here:
    the complete two-phase model will need a more sophisticated positivity
    limiter for phase masses and primitive-variable recovery.
    """

    def __init__(self, space: DGSpace1D, lower: float, upper: float):
        self.space = space
        self.lower = lower
        self.upper = upper
        self.scalar_limiter = SlopeLimiterN(space)

    def __call__(self, alpha_hat: np.ndarray) -> np.ndarray:
        alpha_limited = self.scalar_limiter(alpha_hat)
        alpha_q = self.space.psi @ alpha_limited

        # The correction activates only if the polynomial overshoots.  L2
        # projection retains the expected modal representation for the next RK
        # stage; its small mass change is monitored below.
        if np.any(alpha_q < self.lower) or np.any(alpha_q > self.upper):
            alpha_q = np.clip(alpha_q, self.lower, self.upper)
            alpha_limited = project_to_modal(self.space, alpha_q)
        return alpha_limited


def Lh_driftflux(alpha_modal: list[np.ndarray], t: float, space: DGSpace1D):
    """Assemble the semi-discrete DG RHS for the reduced drift-flux equation."""
    alpha_hat = alpha_modal[0]

    # Modal state with one ghost cell at each pipe boundary.  The left boundary
    # is prescribed (inflow); all characteristic speeds are positive here, so
    # the right boundary is transmissive.
    alpha_t = np.zeros((space.Nldof, space.K + 2))
    alpha_t[:, 1:-1] = alpha_hat

    alpha_inside_left = space.Flkp1[0, :] @ alpha_t[:, 1]
    alpha_inside_right = space.Frk[0, :] @ alpha_t[:, -2]
    alpha_t[0, 0] = 2.0 * alpha_left - alpha_inside_left
    alpha_t[0, -1] = alpha_inside_right

    # Nonlinear physical flux projected with the toolkit superintegration.
    flux_t = FluxProjection(
        space,
        [alpha_t],
        DriftFlux.gas_void_fraction_flux,
        mixture_velocity=j_m,
        distribution_parameter=C0,
        drift_velocity=Vgj,
    )[0]

    # FluxProjection acts only on physical cells; impose the two ghost fluxes
    # explicitly from the ghost states used by the interface Rusanov flux.
    flux_t[0, 0] = DriftFlux.gas_void_fraction_flux(
        alpha_t[0, 0], j_m, C0, Vgj
    )
    flux_t[0, -1] = DriftFlux.gas_void_fraction_flux(
        alpha_t[0, -1], j_m, C0, Vgj
    )

    # A local Rusanov bound is enough for the scalar model.  Each interface
    # takes the maximum |dF/dalpha| from its two adjacent traces.
    alpha_left_trace = space.Flkp1[0, :] @ alpha_t[:, 1:-1]
    alpha_right_trace = space.Frk[0, :] @ alpha_t[:, 1:-1]
    speed_left_trace = np.abs(
        DriftFlux.gas_void_fraction_speed(alpha_left_trace, j_m, C0, Vgj)
    )
    speed_right_trace = np.abs(
        DriftFlux.gas_void_fraction_speed(alpha_right_trace, j_m, C0, Vgj)
    )
    C_local = np.empty(space.K + 1)
    C_local[0] = max(
        abs(DriftFlux.gas_void_fraction_speed(alpha_t[0, 0], j_m, C0, Vgj)),
        speed_left_trace[0],
    )
    C_local[1:-1] = np.maximum(speed_right_trace[:-1], speed_left_trace[1:])
    C_local[-1] = max(speed_right_trace[-1], speed_right_trace[-1])

    volume = space.S.T @ flux_t[:, 1:-1]
    face = rusanov(space, alpha_t, flux_t, C_local)
    rhs_alpha = space.InvM[:, None] * space.J[None, :] ** (-1) * (volume + face)

    return [rhs_alpha]


def shock_speed() -> float:
    """Rankine-Hugoniot velocity for the chosen scalar Riemann problem."""
    flux_left = DriftFlux.gas_void_fraction_flux(alpha_left, j_m, C0, Vgj)
    flux_right = DriftFlux.gas_void_fraction_flux(alpha_right, j_m, C0, Vgj)
    return (flux_right - flux_left) / (alpha_right - alpha_left)


def exact_void_fraction(x: np.ndarray, t: float) -> np.ndarray:
    """Exact entropy solution for this single-shock Riemann test."""
    x_shock = x_jump + shock_speed() * t
    return np.where(x < x_shock, alpha_left, alpha_right)


def element_average_mass(space: DGSpace1D, alpha_hat: np.ndarray) -> float:
    """Domain integral of alpha_g; mode zero is the cell average for Legendre."""
    return float(np.sum(2.0 * space.J * alpha_hat[0, :]))


def main() -> None:
    space = DGSpace1D(K=K, N=N, xmin=0.0, xmax=L, quad_type="GL")
    limiter = VoidFractionLimiter(space, alpha_floor, 1.0 - alpha_floor)

    alpha_initial = np.where(space.xc < x_jump, alpha_left, alpha_right)
    alpha_modal = project_to_modal(space, alpha_initial)
    U_modal = [alpha_modal]

    # CFL uses the maximum characteristic speed over the admissible interval.
    alpha_bounds = np.array([alpha_left, alpha_right])
    lambda_max = float(
        np.max(
            np.abs(
                DriftFlux.gas_void_fraction_speed(alpha_bounds, j_m, C0, Vgj)
            )
        )
    )
    dx = L / K
    dt_cfl = CFL * dx / ((2 * N + 1) * lambda_max)
    nsteps = int(np.ceil(final_time / dt_cfl))
    dt = final_time / nsteps

    mass0 = element_average_mass(space, U_modal[0])
    boundary_mass_rate = float(
        DriftFlux.gas_void_fraction_flux(alpha_left, j_m, C0, Vgj)
        - DriftFlux.gas_void_fraction_flux(alpha_right, j_m, C0, Vgj)
    )
    time_history = [0.0]
    mass_history = [mass0]

    print("=" * 72)
    print("DG drift-flux kinematic-wave test")
    print("Model: alpha_t + d/dx[alpha(C0*j + Vgj(1-alpha))] = 0")
    print(f"K={K}, N={N}, CFL={CFL}, dt={dt:.3e}, steps={nsteps}")
    print(f"j={j_m:.3f} m/s, C0={C0:.3f}, Vgj={Vgj:.3f} m/s")
    print(f"Exact shock speed = {shock_speed():.6f} m/s")
    print("=" * 72)

    t = 0.0
    rhs = lambda state, time: Lh_driftflux(state, time, space)
    report_every = max(1, nsteps // 8)

    for step in range(1, nsteps + 1):
        U_modal = RKSSP54_Step(U_modal, t, dt, rhs, limiter)
        t += dt

        if step % 10 == 0 or step == nsteps:
            time_history.append(t)
            mass_history.append(element_average_mass(space, U_modal[0]))
        if step % report_every == 0 or step == nsteps:
            alpha_now = space.psi @ U_modal[0]
            print(
                f"step {step:5d}/{nsteps}: t={t:.4f} s, "
                f"alpha in [{alpha_now.min():.6f}, {alpha_now.max():.6f}]"
            )

    x_plot = space.xc.T.reshape(-1)
    alpha_dg = (space.psi @ U_modal[0]).T.reshape(-1)
    alpha_exact = exact_void_fraction(x_plot, final_time)
    liquid_dg = 1.0 - alpha_dg
    liquid_exact = 1.0 - alpha_exact

    l1_error = np.mean(np.abs(alpha_dg - alpha_exact))
    l2_error = np.linalg.norm(alpha_dg - alpha_exact) / np.linalg.norm(alpha_exact)
    expected_mass = mass0 + boundary_mass_rate * np.asarray(time_history)
    mass_balance_residual = np.asarray(mass_history) - expected_mass
    print("-" * 72)
    print(f"Relative L2(alpha_g) = {l2_error:.4e}")
    print(f"Mean absolute error  = {l1_error:.4e}")
    print(f"Integral-alpha change = {mass_history[-1] - mass0:+.4e}")
    print(f"Mass-balance residual = {mass_balance_residual[-1]:+.4e}")

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), constrained_layout=True)
    axes[0].plot(x_plot, alpha_exact, "k-", lw=2.0, label="Exact shock")
    axes[0].plot(x_plot, alpha_dg, color="#0072B2", lw=1.4, label="DG (RKSSP54 + Rusanov)")
    axes[0].set_ylabel(r"Gas void fraction $\alpha_g$")
    axes[0].set_ylim(-0.03, 1.03)
    axes[0].set_xlim(0.0, L)
    axes[0].set_xlabel("x [m]")
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="best")
    axes[0].set_title("Reduced drift-flux kinematic wave: gas-liquid slip")

    axes[1].plot(x_plot, liquid_exact, "k-", lw=2.0, label="Exact")
    axes[1].plot(x_plot, liquid_dg, color="#D55E00", lw=1.4, label="DG")
    axes[1].set_ylabel(r"Liquid holdup $1-\alpha_g$")
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].set_xlim(0.0, L)
    axes[1].set_xlabel("x [m]")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="best")

    axes[2].plot(time_history, mass_balance_residual, color="#009E73", lw=1.8)
    axes[2].axhline(0.0, color="k", lw=0.8)
    axes[2].set_xlabel("Time [s]")
    axes[2].set_ylabel(r"Mass-balance residual")
    axes[2].grid(alpha=0.3)

    fig.savefig(OUTPUT_DIR / "driftflux_kinematic_result.png", dpi=180, bbox_inches="tight")
    np.savez(
        OUTPUT_DIR / "driftflux_kinematic_results.npz",
        x=x_plot,
        alpha_dg=alpha_dg,
        alpha_exact=alpha_exact,
        liquid_dg=liquid_dg,
        liquid_exact=liquid_exact,
        time=np.asarray(time_history),
        mass=np.asarray(mass_history),
        expected_mass=expected_mass,
        mass_balance_residual=mass_balance_residual,
        final_time=final_time,
        shock_speed=shock_speed(),
        dt=dt,
        K=K,
        N=N,
        C0=C0,
        mixture_velocity=j_m,
        drift_velocity=Vgj,
    )

    plt.show()


if __name__ == "__main__":
    main()
