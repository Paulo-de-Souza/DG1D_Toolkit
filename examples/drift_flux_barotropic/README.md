# Periodic barotropic drift-flux system

Run from the repository root:

```bash
python -m examples.drift_flux_barotropic.run
```

## Model

The conservative state is $U=[m_l,m_g,G]^T$, where $m_l=\alpha_l\rho_l$, $m_g=\alpha_g\rho_g$, and $G=m_lu_l+m_gu_g$. The source-free model is

$$
\partial_tm_l+\partial_x(m_lu_l)=0,
\qquad
\partial_tm_g+\partial_x(m_gu_g)=0,
$$

$$
\partial_tG+\partial_x\left(m_lu_l^2+m_gu_g^2+p\right)=0.
$$

The closure uses constant liquid density, $p=c_g^2\rho_g$, and the drift relation $u_g=C_0j+V_{gj}$. A smooth gas-rich packet evolves on a
periodic domain $[0,1]$; therefore all three global integrals should be conserved. The parameters are nondimensional study values, not a calibrated
fluid model.

## Discretisation

The driver uses 120 elements, degree 1, Rusanov interfaces, SSPRK(5,4), and an adaptive CFL step. The figure and `.npz` file record initial/final fields and conservation residuals. New output goes to `results/generated/`.
