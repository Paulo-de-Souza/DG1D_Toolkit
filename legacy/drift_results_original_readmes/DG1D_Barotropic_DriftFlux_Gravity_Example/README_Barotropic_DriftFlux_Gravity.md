# Cenário 3: tubo aberto inclinado com Drift-Flux barotrópico

Execute na raiz de `DG1D_Toolkit`:

```bash
python main_driftflux_barotropic_gravity.py
```

O estado e os fluxos são os mesmos do cenário 2,

\[
\mathbf U=[m_l,m_g,G]^{\mathsf T},
\qquad
F(\mathbf U)=
\begin{bmatrix}
m_lu_l\\m_gu_g\\m_lu_l^2+m_gu_g^2+p
\end{bmatrix},
\]

mas a equação de momento recebe o termo de gravidade

\[
S_G=-(m_l+m_g)g^*\sin\theta.
\]

No caso implementado, o tubo sobe na direção positiva de \(x\); portanto,
a gravidade reduz o momento positivo da mistura.

## Condições de contorno e teste

- **Entrada em \(x=0\):** mistura imposta com \(\alpha_g=0.20\) e
  \(j=0.45\).
- **Saída em \(x=L\):** extrapolação transmissiva simples.
- **Inclinação:** \(25^\circ\) ascendente.
- **Estado inicial:** pacote suave rico em gás, centrado em \(x=0.60\),
  transportado rumo à saída.
- **Sensor:** \(x_s=1.0\), no caminho do pacote.

A figura final deve ser lida assim: a fração de gás mostra o deslocamento do
pacote; a pressão deixa de ser uniforme e cresce a montante para sustentar o
escoamento na subida; e as velocidades de líquido, gás e mistura diminuem ao
longo do tubo porque a gravidade se opõe ao movimento.

## Escopo

O objetivo é demonstrar o acoplamento **conservação + slip + pressão +
gravidade + entrada/saída**. Os parâmetros continuam adimensionais. Não há
atrito de parede, energia, transferência de massa ou closures por padrão de
escoamento. As condições de contorno são ghosts simples para fins didáticos;
um caso físico de pesquisa depois deve usar condições características e uma
closure de atrito/regime escolhida para o fluido e a geometria reais.
