# Allen–Cahn: exemplo independente

Este diretório contém uma versão autocontida do teste suave de Allen–Cahn
usado nos scripts do professor Alberto. O `run.py` não importa
`examples.alberto_replicas.common`: toda a montagem do caso fica visível no
próprio exemplo, usando as ferramentas públicas da `dg1d_toolkit`.

## Como executar

Na raiz da toolkit, a forma recomendada é:

```bash
python -m examples.allen_cahn.run --quick
```

Também funciona a chamada direta:

```bash
python examples/allen_cahn/run.py --quick
```

Para a configuração de produção do Alberto:

```bash
python -m examples.allen_cahn.run --diffusion ldg --print-every 1000
```

O caso alternativo com BR2 pode ser executado com:

```bash
python -m examples.allen_cahn.run --diffusion br2 --print-every 1000
```

`--quick` é apenas um teste curto de instalação. Ele não altera os valores
do caso completo. `--print-every` controla as mensagens no terminal;
`--history-every` controla quantos snapshots são guardados para o mapa 2D e o
GIF. Por padrão, o programa escolhe aproximadamente 200 frames. Para gerar
somente o arquivo numérico, use `--no-plot --no-gif`.

## Problema e configuração

O teste resolve

$$
\partial_t u = \mu\left(\beta u + \varepsilon^2\partial_{xx}u\right),
\qquad x\in[0,4\pi],
$$

com condição periódica e

$$
u(x,0)=\cos(x/2),\qquad
\mu=1,\quad \beta=0.5,\quad \varepsilon=\sqrt{6}.
$$

Como $\beta-\varepsilon^2/4=-1$, a solução exata usada na comparação é

$$
u(x,t)=e^{-t}\cos(x/2).
$$

A configuração completa preserva os valores efetivos do laço do Alberto:

| parâmetro | valor |
| --- | ---: |
| elementos `K` | 20 |
| grau `N` | 3 |
| quadratura | Gauss–Legendre |
| projeção linear / não linear | `P=2N` / `nlP=4N` |
| passo `dt` | `2.5e-4` |
| passos executados | 20 000 |
| tempo efetivo | 5.0 |

O fluxo de reação é projetado com `FluxProjection`. O termo difusivo usa
`diffusion_rhs` com LDG por padrão; `--diffusion br2` seleciona a formulação
BR2 disponibilizada pela toolkit. A evolução temporal usa `RK4_Step`.

## Saídas

Os arquivos são gravados em `results/generated/`:

- `allen_cahn_ldg.npz` ou `allen_cahn_br2.npz`: solução final, histórico,
  coordenadas, tempos, parâmetros e erro infinito;
- `*_final.png`: condição inicial, solução DG final e solução exata;
- `*_evolution.png`: mapa 2D de $u(x,t)$;
- `*_animation/*.gif`: animação da solução;
- `*_animation/frame_*.png`: frames intermediários usados pelo GIF.

Os resultados são regeneráveis e podem ser ignorados pelo Git conforme o
`.gitignore` da raiz.
