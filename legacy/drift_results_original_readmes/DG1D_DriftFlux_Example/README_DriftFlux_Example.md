# Exemplo DG 1D: onda cinemática Drift-Flux

Execute, dentro da pasta `DG1D_Toolkit`:

```bash
python main_driftflux_kinematic.py
```

O exemplo resolve a primeira redução verificável de um modelo bifásico:

\[
\frac{\partial \alpha_g}{\partial t}+
\frac{\partial}{\partial x}
\left\{\alpha_g\left[C_0j+V_{gj}(1-\alpha_g)\right]\right\}=0.
\]

`alpha_g` é a fração de vazio do gás; `j` é a velocidade superficial da
mistura; `C0` é o parâmetro de distribuição; `Vgj` representa o drift/slip do
gás. Portanto, mesmo sendo uma única PDE, não é a advecção escalar comum: a
velocidade depende da fração de gás por meio da lei de drift-flux.

## O que usa da toolkit

- `DGSpace1D`: malha, base modal de Legendre, massa, rigidez e lift;
- `FluxProjection`: projeção L2 superintegrada do fluxo não linear;
- `rusanov`: fluxo numérico local, com `dF/dalpha` como velocidade de onda;
- `RKSSP54_Step` e `SlopeLimiterN`: avanço temporal e estabilização de choque.

O problema de Riemann adotado tem solução analítica de um choque. Assim, o
gráfico `driftflux_kinematic_result.png` confronta diretamente DG e solução
exata, além de monitorar o resíduo do balanço global de `alpha_g`. Como o tubo
tem entrada e saída, a integral de `alpha_g` não é constante; ela deve variar
exatamente conforme o fluxo de gás nas duas fronteiras.

## Escopo físico deliberado

Este ainda **não** é o sistema completo da figura do ALFASim. A imagem contém
duas continuidades, dois balanços de momento e uma energia global — modelo
two-fluid com muitas relações de fechamento. O código reduzido congela
pressão, momento da mistura, energia, atrito e gravidade para testar, antes de
tudo, o bloco que será reutilizado: conservação DG de uma variável de fase com
fluxo de drift e controle de `0 <= alpha_g <= 1`.

O próximo nível natural é `U = [m_l, m_g, G]^T`, com recuperação de primitivas
e uma equação barotrópica. Só depois fazem sentido energia, fricção, gravidade
e closures dependentes de regime de escoamento.
