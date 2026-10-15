# Water hammer with gradual valve closure

Run from the repository root:

```bash
python -m examples.water_hammer_gradual_valve.run_dg
python -m examples.water_hammer_gradual_valve.run_moc
python -m examples.water_hammer_gradual_valve.compare_dg_moc
```

## Model and conditions

The linear water-hammer model used in the scripts is

\[
\partial_tH+\frac{a^2}{g}\partial_xV=0,
\qquad
\partial_tV+g\partial_xH=-\frac{f}{2D}V|V|.
\]

The pipe has \(L=300\,\mathrm{m}\), wave speed \(a=1000\,\mathrm{m/s}\),
diameter \(D=0.05\,\mathrm{m}\), friction factor \(f=0.015\), upstream
reservoir head \(H(0,t)=50\,\mathrm{m}\), and initially uniform velocity
\(V_0=0.412\,\mathrm{m/s}\). At the downstream valve,

\[
V(L,t)=V_0\max\left(0,1-t/t_c\right),\qquad t_c=0.05\,\mathrm{s}.
\]

The DG result is compared with a companion method-of-characteristics (MOC)
calculation at 50 m and 200 m sensors. Precomputed arrays and figures are in
`results/reference_data/`; new files go to `results/generated/`.
