# Reduced drift-flux kinematic wave

Run from the repository root:

```bash
python -m examples.drift_flux_kinematic.run
```

## Model

This first two-phase step advances only the gas void fraction:

\[
\partial_t\alpha_g+
\partial_x\left[\alpha_g\left(C_0j+V_{gj}(1-\alpha_g)\right)\right]=0.
\]

It uses \(L=1\,\mathrm{m}\), mixture superficial velocity
\(j=0.8\,\mathrm{m/s}\), \(C_0=1.2\) and \(V_{gj}=0.35\,\mathrm{m/s}\).
The Riemann data are \(\alpha_g=0.08\) left of \(x=0.4\) and
\(\alpha_g=0.62\) to the right. The left boundary is prescribed and the
right boundary is transmissive because the chosen characteristics leave the
domain there.

## Verification

For the stated data the solution is a single entropy shock. The script uses
the Rankine--Hugoniot speed to construct an exact solution, reports L1/L2
error and a flux-based global mass-balance residual. It uses 160 elements,
degree 2, Rusanov flux, SSPRK(5,4), and a void-fraction limiter enforcing
\(0\leq\alpha_g\leq1\). Output is saved in `results/generated/`; the prior
reference output is in `results/reference_data/`.
