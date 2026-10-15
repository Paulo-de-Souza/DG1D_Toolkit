# Inviscid Burgers equation

Run from the repository root:

```bash
python -m examples.burgers_inviscid.run
```

## Model

The example solves the nonlinear conservation law

\[
\partial_t u + \partial_x\left(\frac{u^2}{2}\right)=0,
\qquad x\in[0,1],\quad t\in[0,0.5].
\]

The initial state is

\[
u(x,0)=0.5+\sin(2\pi x).
\]

The current driver applies its left ghost state through
`contorno_dirichlet`; its right ghost is set to the same prescribed state.
This is retained as an exploratory limiter demonstration rather than a
benchmark with an analytical boundary solution.

## Discretisation and output

The driver uses 250 elements, degree 4, Lax--Friedrichs flux,
SSPRK(10,4), and the hierarchical limiter. It writes a GIF and individual
frames to `results/generated/animation/`. The original animation is preserved
in `assets/animations/burgers_inviscid/`.
