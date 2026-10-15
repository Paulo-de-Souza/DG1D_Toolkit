# Sod shock tube: 1-D Euler equations

Run from the repository root:

```bash
python -m examples.euler_sod.run
```

## Model and data

The compressible Euler system is

\[
\partial_t
\begin{bmatrix}\rho\\\rho u\\E\end{bmatrix}
+\partial_x
\begin{bmatrix}
\rho u\\
\rho u^2+p\\
u(E+p)
\end{bmatrix}=0,
\qquad
p=(\gamma-1)\left(E-\frac{(\rho u)^2}{2\rho}\right),
\]

with \(\gamma=1.4\), \(x\in[0,1]\) and final time \(t=0.2\). The initial
Riemann data are the standard Sod states:

\[
(\rho,u,p)_L=(1,0,1),\qquad(\rho,u,p)_R=(0.125,0,0.1),
\]

separated at \(x=0.5\). The boundary ghost states impose the corresponding
left and right primitive states.

## Discretisation and output

The driver uses 180 elements, degree 3, local Rusanov flux, SSPRK(5,4), and a
slope limiter. It computes an exact Riemann reference and saves density,
velocity and pressure comparisons to `results/generated/`.
