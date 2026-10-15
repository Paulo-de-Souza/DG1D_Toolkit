# Inclined open-pipe barotropic drift-flux model

Run from the repository root:

```bash
python -m examples.drift_flux_inclined.run
```

## Model

This example uses the same three-equation barotropic drift-flux system as the
periodic case, but adds gravity to mixture momentum:

\[
\partial_tG+\partial_x\left(m_lu_l^2+m_gu_g^2+p\right)
=-(m_l+m_g)g^*\sin\theta.
\]

The pipe rises in positive \(x\) with \(\theta=25^\circ\); hence gravity
opposes the positive flow. At the inlet, the driver prescribes
\(\alpha_g=0.20\) and \(j=0.45\). At the outlet it applies a simple
transmissive ghost state. The initial condition is a smooth gas-rich packet
centred at \(x=0.60\), and a virtual sensor lies at \(x=1.0\).

## Scope

The model demonstrates conservation, slip, pressure, gravitational forcing
and open boundaries using nondimensional parameters. The outlet condition is
deliberately simple; use the characteristic-boundary example for the next
level of boundary treatment. Output is written to `results/generated/` and a
prior result is retained in `results/reference_data/`.
