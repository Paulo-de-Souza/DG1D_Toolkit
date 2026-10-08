# Scope and validation status

The examples serve different purposes and should not be assigned the same
validation status.

- **Burgers and Euler:** numerical-method demonstrations with analytical or
  standard Riemann-problem references.
- **Water hammer:** comparison between a DG implementation and a companion
  method-of-characteristics calculation under the assumptions encoded in each
  example.
- **Kinematic drift-flux:** reduced scalar model with an exact shock solution.
- **Barotropic drift-flux:** conservation and boundary-condition demonstrations
  with nondimensional study parameters.

The barotropic drift-flux cases do not yet include calibrated fluid EOSs,
wall-friction closures, energy conservation, mass transfer, surface tension,
or flow-regime-dependent closures. They must therefore not be interpreted as
predictive oil-and-gas pipeline simulations.
