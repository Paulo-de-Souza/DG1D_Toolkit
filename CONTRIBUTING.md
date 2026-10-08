# Contributing

This repository is used by a research group. Keep contributions small,
reproducible and explicit about their numerical scope.

1. Create a branch for one purpose.
2. Keep reusable operations in `dg1d_toolkit/` and case-specific physics in an
   `examples/<case>/` directory.
3. State the PDE, initial/boundary conditions, numerical flux, limiter and
   reference solution in the example README.
4. Do not commit caches or newly generated output files by default.
5. Add a regression test when changing the numerical kernel.
6. Report Python and dependency versions with results intended for comparison.

Avoid presenting a numerical demonstration as physical validation unless the
necessary closure models, parameters and validation data are documented.
