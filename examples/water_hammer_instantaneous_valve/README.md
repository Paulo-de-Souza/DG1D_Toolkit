# Water hammer with instantaneous valve closure

Run from the repository root:

```bash
python -m examples.water_hammer_instantaneous_valve.run_dg
python -m examples.water_hammer_instantaneous_valve.run_moc
python -m examples.water_hammer_instantaneous_valve.compare_dg_moc
```

## Model and conditions

This lossless variant solves

$$
\partial_tH+\frac{a^2}{g}\partial_xV=0,
\qquad
\partial_tV+g\partial_xH=0,
$$

in an $800\,\mathrm{m}$ pipe with $a=1000\,\mathrm{m/s}$, upstream head $H(0,t)=20\,\mathrm{m}$, and initial velocity
$V_0=0.15\,\mathrm{m/s}$. The valve boundary switches instantly to $V(L,t)=0$ for $t>0$.

The DG driver uses 50 degree-one elements and a slope limiter. The MOC driver and comparison script provide a lossless reference based on the Joukowsky head jump $\Delta H=aV_0/g$. Existing data and figures are under `results/reference_data/`; regenerated output stays in `results/generated/`.
