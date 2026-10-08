# Inclined drift-flux model with characteristic boundaries

Run from the repository root:

```bash
python -m examples.drift_flux_characteristic.run
```

## Model

The conservation laws and gravity source are the same as in
`../drift_flux_inclined`:

$$
\partial_tU+\partial_xF(U)=
\begin{bmatrix}0\\ 
0 \\
-(m_l+m_g)g^* \sin \theta
\end{bmatrix},
\qquad U=[m_l,m_g,G]^T.
$$

The code evaluates a numerical flux Jacobian $A=\partial F/\partial U$ and decomposes the boundary jump into local characteristic components. At
$x=0$, only components with positive characteristic speed receive the prescribed inlet data $(\alpha_g,j)=(0.20,0.45)$. At $x=L$, only the
incoming negative-speed components receive a target barotropic pressure; outgoing components are kept from the interior.

## Scope

The scenario uses the same gas-rich packet, 25-degree uphill pipe and nondimensional closure as the inclined example. The characteristic boundary
condition is locally linearised and is intended as a numerical-method study, not a final physical boundary model. Generated plots and arrays are placed in
`results/generated/`; the original reference output is retained alongside it.
