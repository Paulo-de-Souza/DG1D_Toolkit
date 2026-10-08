# Viscous Burgers equation

Run from the repository root:

```bash
python -m examples.burgers_viscous.run
```

## Model

The equation is

$$
\partial_t u + \partial_x\left(\frac{u^2}{2}\right)
=\nu\,\partial_{xx}u,
\qquad x\in[0,2\pi],\quad t\in[0,0.5],
$$

with $\nu=0.07$. The initial field and the Dirichlet boundary trace at $x=0$ are built from the same Cole--Hopf analytical expression in the
driver; the right ghost uses the matching prescribed value. This allows the numerical field to be compared with the analytical field at several times.

## Discretisation and output

The implementation uses 20 elements, polynomial degree 10, BR1 for the diffusive contribution, Lax--Friedrichs for convection and SSPRK(10,4).
The comparison figure is written to `results/generated/`.
