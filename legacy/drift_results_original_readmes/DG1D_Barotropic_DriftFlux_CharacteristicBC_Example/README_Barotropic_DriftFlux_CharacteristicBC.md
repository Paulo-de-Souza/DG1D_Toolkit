# Cenário 3b: tubo inclinado com condições de contorno características

Execute na raiz de `DG1D_Toolkit`:

```bash
python main_driftflux_barotropic_gravity_characteristic.py
```

Este caso usa o mesmo sistema barotrópico de três equações

\[
\mathbf U=[m_l,m_g,G]^{\mathsf T},
\]

mais o termo de gravidade na equação de momento,

\[
S_G=-(m_l+m_g)g^*\sin\theta.
\]

## O que muda em relação ao cenário de ghosts simples

Em cada fronteira, o código calcula a Jacobiana numérica

\[
\mathbf A(\mathbf U)=\frac{\partial\mathbf F}{\partial\mathbf U}
=\mathbf R\mathbf\Lambda\mathbf R^{-1}.
\]

Na **entrada** \(x=0\), somente características com \(\lambda>0\), que entram
no domínio, recebem os dados externos de \(\alpha_g\) e \(j\). As ondas com
\(\lambda<0\), que viajam para fora pela entrada, são mantidas da solução
interna.

Na **saída** \(x=L\), somente características com \(\lambda<0\) recebem a
contribuição do estado externo. O estado externo especifica uma pressão
barotrópica \(p_\mathrm{out}\); as características que saem à direita
permanecem livres. Assim, a fronteira não tenta impor as três variáveis de uma
vez e reduz as reflexões artificiais.

## O que observar

- O pacote gasoso parte de \(x\approx0.60\) e alcança o sensor em
  \(x_s=1.0\).
- A pressão é maior perto da entrada e diminui em direção à saída, coerente
  com escoamento ascendente.
- A pressão próxima da saída permanece próxima de \(p_\mathrm{out}\).
- O gás continua mais rápido que o líquido, pois a closure de drift é mantida.

Os parâmetros são adimensionais. Esta implementação é uma BC característica
linearizada localmente, adequada ao estudo inicial do esquema DG. A validação
física posterior ainda pede uma EOS e closures compatíveis com o fluido/regime
reais, além de atrito de parede.
