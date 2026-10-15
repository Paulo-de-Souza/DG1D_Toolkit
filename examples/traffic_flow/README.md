# Traffic Flow: exemplo independente

Este diretório contém uma versão autocontida do teste `Green-Light` de
Traffic Flow usado nos scripts do professor Alberto. O `run.py` monta o caso
diretamente com a API da `dg1d_toolkit`, sem depender do `common.py` das
réplicas agrupadas.

## Como executar

Na raiz da toolkit, use:

```bash
python -m examples.traffic_flow.run --quick
```

A chamada direta também é suportada:

```bash
python examples/traffic_flow/run.py --quick
```

Para reproduzir a configuração completa:

```bash
python -m examples.traffic_flow.run --print-every 500 --history-every 20
```

O `--print-every` apenas controla a frequência das mensagens no terminal.
O `--history-every` controla a frequência dos snapshots usados no mapa 2D e
no GIF; por padrão, são selecionados aproximadamente 200 frames. Use
`--no-plot --no-gif` para executar somente a integração e salvar o `.npz`.

## Modelo e configuração

O modelo escalar é

$$
\partial_t u + \partial_x f(u)=0,
\qquad
f(u)=V_m\left(u-\frac{u^2}{u_m}\right),
\qquad x\in[0,4].
$$

O teste usa $V_m=1$, $u_m=10$, condição inicial

$$
u(x,0)=
\begin{cases}
\frac{u_m}{2}x, & \text{na metade esquerda da malha},\\
0, & \text{na metade direita da malha},
\end{cases}
$$

e condições de Green-Light $u(0,t)=u(4,t)=0$. A discretização mantém os
valores efetivos do laço do Alberto:

| parâmetro | valor |
| --- | ---: |
| elementos `K` | 39 |
| grau `N` | 9 |
| quadratura | Gauss–Legendre |
| projeção linear / não linear | `P=2N` / `nlP=3N` |
| passo `dt` | `2.5e-4` |
| passos executados | 4 000 |
| tempo efetivo | 1.0 |
| velocidade máxima `Vm` | 1.0 |
| densidade máxima `um` | 10.0 |

O fluxo físico é projetado com `FluxProjection`, o fluxo convectivo de
interface é Rusanov, e a captura de choque usa o sensor Persson/Warburton com
`sensor_constant=1`, `viscosity_factor=0.5`, perfil linear e difusão LDG. A
integração no tempo usa `RKSSP54_Step`.

## Saídas

Os arquivos são gravados em `results/generated/`:

- `traffic_flow.npz`: solução final, histórico, coordenadas e diagnóstico do
  sensor (`sensor_activation` e `sensor_viscosity`);
- `traffic_flow_final.png`: condição inicial e perfil final;
- `traffic_flow_evolution.png`: mapa 2D da densidade ao longo do tempo;
- `traffic_flow_animation/traffic_flow.gif`: animação da solução;
- `traffic_flow_animation/frame_*.png`: frames intermediários usados pelo GIF.

Os resultados são regeneráveis e podem ser ignorados pelo Git conforme o
`.gitignore` da raiz.
