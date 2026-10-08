# Migration map from the original folder

The code and result files were reorganised for reproducible GitHub use; no
meaningful source, result array or animation was intentionally discarded.

| Original location | New location |
| --- | --- |
| `main_inv_burgers_teste.py` | `examples/burgers_inviscid/run.py` |
| `main_viscoso_teste.py` | `examples/burgers_viscous/run.py` |
| `main_euler_teste.py` | `examples/euler_sod/run.py` |
| `main_euler_teste_v2.py` | `examples/euler_sod_animation/run.py` |
| `main_waterhammer.py` | `examples/water_hammer_gradual_valve/run_dg.py` |
| `main_waterhammer_v2.py` | `examples/water_hammer_instantaneous_valve/run_dg.py` |
| `WaterHammer_Studies/moc_waterhammer*.py` | matching water-hammer example `run_moc.py` |
| `main_driftflux_*.py` | matching `examples/drift_flux_*/run.py` |
| `dg1d_toolkit/barotropic_driftflux_v2.py` | `dg1d_toolkit/barotropic_driftflux_characteristic.py` |
| `burgers_animation/`, `euler_animation/`, water-hammer GIF folders | `assets/animations/` |
| `drift_results/` arrays and figures | matching `examples/*/results/reference_data/` |

The former drift-flux READMEs are retained in
`legacy/drift_results_original_readmes/` for traceability. Compiled Python
cache directories were excluded because they are machine-specific artifacts.
