# Réplicas dos testes do Alberto

Esta pasta reúne sete exemplos portados para a arquitetura da `DG1D_Toolkit`.
Eles não são uma cópia linha a linha dos notebooks/scripts antigos: preservam
as equações, condições iniciais, fluxos, parâmetros de sensor e escolhas de
estabilização, mas usam `DGSpace1D`, `FluxProjection`, os fluxos numéricos e os
integradores da toolkit.

## Organização desta pasta

Esta pasta possui uma organização um pouco diferente das demais pastas de
`examples`. Nos outros exemplos, normalmente cada diretório resolve um único problema e contém praticamente toda a lógica necessária para sua execução.

Aqui, entretanto, estão reunidas várias réplicas dos testes originalmente desenvolvidos pelo Alberto. Cada subpasta representa um problema físico diferente, mas todos os casos utilizam uma infraestrutura comum:

```text
examples/alberto_replicas/
├── common.py
├── run_all_quick.py
├── allen_cahn/
│   └── run.py
├── cahn_hilliard/
│   └── run.py
├── shallow_water/
│   └── run.py
├── shock_density/
│   └── run.py
├── shu_osher/
│   └── run.py
├── linear_advection/
│   └── run.py
└── traffic_flow/
    └── run.py

## Mapa dos casos

| Exemplo | Fonte principal no ZIP do Alberto | Estratégia usada nesta réplica |
|---|---|---|
| `allen_cahn` | `Allen-Cahn-1D-Modal-DG.py` e `...-BR2.py` | reação suave + difusão LDG; `--diffusion br2` exercita BR2 |
| `cahn_hilliard` | `Cahn-Hilliard-1D-Modal-DG.py` | potencial não linear projetado + operador dividido LDG/BR2 |
| `shallow_water` | `Shallow-Water-1D-Modal-Gen-Slope-Limiter.py` | fluxo de águas rasas + Rusanov + `SlopeLimiterN` |
| `shock_density` | `Shock-Density-1D-Modal-Sub-cell-shock-capturing.py` | Euler + sensor Persson/Warburton + viscosidade artificial LDG |
| `shu_osher` | `Shu-Osher-1D-Modal-Sub-cell-shock-capturing.py` | Euler + captura modal; estado refletido parcial à direita |
| `linear_advection` | `Linear-Advection-1D-Modal-Subcell-shock-cap.py` | degrau advectado + sensor + perfil compacto |
| `traffic_flow` | `Traffic-Flow-1D-Modal-Sub-cell-shock-cap.py` | fluxo LWR Green-Light + sensor + LDG artificial |

## Como executar os exemplos curtos

Na raiz da toolkit:

```bash
PYTHONPATH=. python examples/alberto_replicas/allen_cahn/run.py --quick
PYTHONPATH=. python examples/alberto_replicas/allen_cahn/run.py --quick --diffusion br2
PYTHONPATH=. python examples/alberto_replicas/cahn_hilliard/run.py --quick
PYTHONPATH=. python examples/alberto_replicas/shallow_water/run.py --quick
PYTHONPATH=. python examples/alberto_replicas/shock_density/run.py --quick
PYTHONPATH=. python examples/alberto_replicas/shu_osher/run.py --quick
PYTHONPATH=. python examples/alberto_replicas/linear_advection/run.py --quick
PYTHONPATH=. python examples/alberto_replicas/traffic_flow/run.py --quick
```

Ou todos os smoke tests de uma vez:

```bash
PYTHONPATH=. python examples/alberto_replicas/run_all_quick.py
```

Cada execução salva um `.npz` e uma figura `.png` em `results/` dentro do
respectivo exemplo e imprime faixas dos campos, número de passos e se o estado
permaneceu finito. A figura recebe o mesmo nome do arquivo numérico; por
exemplo, `shock_density.npz` vem acompanhado de `shock_density.png`.
Use `--output caminho.npz` para escolher outro destino ou `--no-plot` para
desativar as figuras.

Para acompanhar a execução e salvar mapas espaço-temporais, use, por exemplo:

```bash
PYTHONPATH=. python examples/alberto_replicas/traffic_flow/run.py \
  --print-every 500 --history-every 500
```

Além de `traffic_flow.png`, isso cria `traffic_flow_evolution.png`. O mesmo
vale para os demais casos. `--print-every` controla apenas o terminal; já
`--history-every` controla os snapshots usados no gráfico de evolução.

## Configurações de referência

Sem `--quick`, cada exemplo usa os valores efetivos do laço executado nos
scripts do Alberto, inclusive o número manual de passos quando ele é diferente
do tempo nominal declarado no arquivo:

| Exemplo | K | N | dt | passos | tempo efetivo |
|---|---:|---:|---:|---:|---:|
| Allen-Cahn | 20 | 3 | 0.00025 | 20000 | 5.0 |
| Cahn-Hilliard | 40 | 1 | 0.0000022 | 99000 | 0.2178 |
| Shallow-Water | 20 | 8 | 0.0002 | 5000 | 1.0 |
| Shock-Density | 80 | 5 | 0.000055 | 32728 | 1.80004 |
| Shu-Osher | 120 | 5 | 0.000003 | 100000 | 0.3 |
| Linear Advection | 12 | 10 | 0.002 | 410 | 0.82 |
| Traffic Flow | 39 | 9 | 0.00025 | 4000 | 1.0 |

Os exemplos de Euler também possuem `--reference`, mantido como um alias
explícito para a configuração de produção correspondente:

```bash
PYTHONPATH=. python examples/alberto_replicas/shock_density/run.py --reference
PYTHONPATH=. python examples/alberto_replicas/shu_osher/run.py --reference
```

Essas execuções podem ser longas: são os testes de produção, não smoke tests.
`--quick` reduz malha, tempo e número de passos somente para uma verificação
rápida da instalação; por isso uma saída com `passos=20` é esperada apenas
quando esse sinalizador foi usado.

Também é possível sobrescrever qualquer configuração sem editar o script:
`--K`, `--N`, `--t-end`, `--dt` e `--steps`. Quando `--steps` é informado, ele
tem precedência e reproduz exatamente o número de iterações desejado.

## Como executar os exemplos completos

Na raiz da toolkit, execute cada caso individualmente. Essas configurações
usam os valores efetivos dos scripts do Alberto, e não o modo reduzido
`--quick`.

```bash
python examples/alberto_replicas/allen_cahn/run.py --print-every 1000 --history-every 500

python examples/alberto_replicas/allen_cahn/run.py --diffusion br2 --print-every 1000 --history-every 500

python examples/alberto_replicas/cahn_hilliard/run.py --print-every 5000 --history-every 1000

python examples/alberto_replicas/linear_advection/run.py --print-every 50 --history-every 25

python examples/alberto_replicas/shallow_water/run.py --print-every 500 --history-every 100

python examples/alberto_replicas/shock_density/run.py --print-every 2000 --history-every 500

python examples/alberto_replicas/shu_osher/run.py --print-every 5000 --history-every 1000

python examples/alberto_replicas/traffic_flow/run.py --print-every 500 --history-every 100


## Observações de fidelidade

* O mecanismo chamado “sub-cell shock capturing” nos scripts ativos do Alberto
  é, na prática, um sensor modal que ativa viscosidade artificial DG. É isso que
  os exemplos reproduzem; o utilitário opcional de médias subcelulares continua
  disponível na toolkit, mas não substitui a evolução DG.
* O Cahn–Hilliard original lê `icr_u.npy` e tem uma formulação de energia livre
  acoplada a dois parâmetros de ordem. Para tornar o exemplo distribuível e
  reproduzível, esta versão usa uma perturbação determinística e a forma
  clássica de Cahn–Hilliard dividida, mantendo a montagem LDG e a projeção
  modal. A diferença está explicitamente isolada no arquivo do exemplo.
* A projeção modal desta réplica usa a orientação correta dos pesos da
  quadratura (`weights[None, :] * basis.T`). O `SlopeLimiterN` atual ainda
  pressupõe `nip=N+1`, por isso a réplica de shallow-water usa GL; o código do
  Alberto usa GLL, mas a física, a IC e o limitador são os mesmos.
* Os arquivos dependem dos módulos de captura adicionados anteriormente:
  `dg1d_capture_shock.py`, `dg1d_subcell.py` e os fluxos LDG/BR2 em
  `numericalfluxes.py`. O pacote de exemplos inclui uma cópia atualizada de
  `dg1d_subcell.py` que preserva ghosts prescritos em problemas não periódicos.
