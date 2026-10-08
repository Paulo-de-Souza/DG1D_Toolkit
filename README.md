# DG1D Toolkit

A research and teaching toolkit for one-dimensional discontinuous Galerkin
(DG) discretisations of conservation laws and selected multiphase-flow
reductions. The repository provides a compact modal DG kernel based on
Jacobi/Legendre polynomials together with self-contained numerical examples.

> **Status.** This is an actively developed research codebase. The drift-flux
> models are numerical prototypes for verification and method development;
> they are not calibrated or validated for field-scale engineering prediction.

## Included capabilities

- modal DG space, quadrature, differentiation matrices, lifting operators and
  nonlinear flux projection;
- Lax--Friedrichs, Rusanov and BR1 numerical fluxes;
- explicit Euler, RK4 and SSP Runge--Kutta integrators;
- slope and hierarchical limiters;
- inviscid/viscous Burgers, the Sod Euler Riemann problem and water-hammer
  examples;
- a staged drift-flux sequence: kinematic wave, barotropic three-equation
  model, inclined pipe with gravity, and characteristic boundary conditions.

## Installation

Create an isolated environment, clone the repository, and install the runtime
dependencies:

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Python 3.10 or newer is required. Running the examples with `python -m` from
the repository root keeps imports and output paths portable.

## Examples

| Example | Model / purpose | Command |
| --- | --- | --- |
| Inviscid Burgers | nonlinear scalar conservation law and limiter animation | `python -m examples.burgers_inviscid.run` |
| Viscous Burgers | convection--diffusion with an analytical reference | `python -m examples.burgers_viscous.run` |
| Euler / Sod | compressible Euler Riemann problem | `python -m examples.euler_sod.run` |
| Water hammer: gradual valve | DG and MOC, frictional transient | `python -m examples.water_hammer_gradual_valve.run_dg` |
| Water hammer: instantaneous valve | lossless closure and analytical trace | `python -m examples.water_hammer_instantaneous_valve.run_dg` |
| Kinematic drift-flux | scalar void-fraction shock with exact solution | `python -m examples.drift_flux_kinematic.run` |
| Barotropic drift-flux | periodic three-equation conservative system | `python -m examples.drift_flux_barotropic.run` |
| Inclined drift-flux | open pipe with gravity source | `python -m examples.drift_flux_inclined.run` |
| Characteristic drift-flux | open pipe with characteristic inlet/outlet | `python -m examples.drift_flux_characteristic.run` |

Every example has a local `README.md` that states the model, conditions,
parameters and outputs. New outputs are written into
`examples/<case>/results/generated/`; versioned figures and data are in
`results/reference_data/` when available.

## Repository layout

```text
dg1d_toolkit/  reusable DG kernel and physical closures
examples/      independently runnable numerical studies
docs/          equations, scope and references
assets/        retained animations from the original experiments
tests/         lightweight regression tests for the numerical kernel
legacy/        migration notes and historical READMEs
```

## Reproducibility and citation

The reference arrays/figures distributed here preserve the earlier study
outputs. Regenerate them only after recording the Python and package versions
used. Please cite the repository using `CITATION.cff`; see `docs/references.md`
for the methodological literature that should accompany research use.

## License and contributions

Released under the [MIT License](LICENSE). Contributions are welcome through
small, focused pull requests; see [CONTRIBUTING.md](CONTRIBUTING.md).
