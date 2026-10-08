# Sod shock tube: animation variant

Run from the repository root:

```bash
python -m examples.euler_sod_animation.run
```

This is the animation-oriented companion to `../euler_sod`. It solves the same Euler equations, gas constant $\gamma=1.4$, Sod initial data and boundary
states. It stores the transient pressure field every 80 time steps and writes a pressure GIF to `results/generated/pressure_animation/`.

The historical GIFs for density, momentum, energy and pressure are retained in `assets/animations/euler_sod/`. For the clean analytical comparison, use the primary `euler_sod` example.
