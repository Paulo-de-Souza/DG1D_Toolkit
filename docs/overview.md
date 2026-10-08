# Numerical formulation

For a one-dimensional conservation law

$$
\partial_t u + \partial_x f(u) = s(u,x,t),
$$

the toolkit represents the solution independently in every element using a modal Jacobi/Legendre basis. The weak element equations use precomputed mass,
stiffness and lifting operators. Nonlinear physical fluxes are evaluated on a superintegration grid and projected back to modal coefficients.

At interfaces, the examples use either local Lax--Friedrichs/Rusanov fluxes or BR1 for the viscous auxiliary flux. Time advancement is explicit, with
Euler, RK4, SSPRK(5,4) and SSPRK(10,4) implementations. Discontinuous cases activate the appropriate limiter in the individual example.

The library deliberately separates the generic DG operations from physical closures. A new case should normally provide only its flux, source, boundary
conditions and a short driver that assembles its semi-discrete right-hand side.
