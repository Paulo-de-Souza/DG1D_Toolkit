# Exemplo DG 1D: Drift-Flux barotrópico com três equações

Execute da raiz de `DG1D_Toolkit`:

```bash
python main_driftflux_barotropic.py
```

O estado conservativo é

\[
\mathbf U=[m_l,m_g,G]^{\mathsf T},
\]

com

\[
m_l=\alpha_l\rho_l,\qquad
m_g=\alpha_g\rho_g,\qquad
G=m_lu_l+m_gu_g.
\]

As três equações são

\[
\partial_t m_l+\partial_x(m_lu_l)=0,
\]

\[
\partial_t m_g+\partial_x(m_gu_g)=0,
\]

\[
\partial_tG+\partial_x(m_lu_l^2+m_gu_g^2+p)=0.
\]

O fechamento do primeiro nível é

\[
\rho_l=\mathrm{constante},\qquad p=c_g^2\rho_g,
\]

\[
u_g=C_0j+V_{gj},\qquad j=\alpha_lu_l+\alpha_gu_g,
\]

mais \(\alpha_l+\alpha_g=1\). A recuperação de primitivas calcula
\(\alpha_g,\rho_g,p,j,u_l,u_g\) diretamente de \([m_l,m_g,G]\), sem um solve
não linear por ponto.

## Caso de teste

O script coloca um pacote suave, inicialmente rico em gás, em um tubo
periódico sem termos fonte. Isso permite verificar as integrais de \(m_l\),
\(m_g\) e \(G\) sem mistura com incertezas de condições de contorno. A figura
final apresenta fração de vazio, pressão, velocidades de líquido/gás e os
resíduos de conservação.

O fluxo físico usa `FluxProjection`; as interfaces usam Rusanov. O módulo
`barotropic_driftflux.py` calcula a velocidade de Rusanov pelo raio espectral
de uma Jacobiana numérica \(\partial F/\partial U\).

## Limites deliberados

Os parâmetros são **adimensionais**, escolhidos para verificação numérica
estável. Não há gravidade, atrito, energia, transferência de massa, tensão
interfacial ou seleção de regime. A pressão do gás é apenas barotrópica, não
uma EOS de óleo/gás calibrada.

O próximo degrau é acrescentar gravidade e atrito de mistura mantendo esse
núcleo conservativo e a recuperação de primitivas.
