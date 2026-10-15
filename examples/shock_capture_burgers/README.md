# Burgers: limitador versus captura por viscosidade artificial

Este estudo reproduz, na arquitetura modular da toolkit, a ideia dos
arquivos `Inviscid-Burgers-1D-Modal-Sub-cell-shock-cap.py` e
`Euler-1D-Modal-Sub-cell-shock-capturing.py` do Alberto.

Ele roda duas simulações com a mesma projeção, fluxo Lax–Friedrichs antigo e
integrador SSPRK(10,4):

1. `HierarchicalLimiter`, já existente na toolkit;
2. sensor Persson/Warburton + `epssmoother` + stiffness modificado + LDG.

Execute a partir da raiz da toolkit:

```bash
python examples/shock_capture_burgers/run.py
```

Parâmetros úteis:

```bash
python examples/shock_capture_burgers/run.py \
    --K 120 --N 4 --t-end 0.25 --dt 0.0005 --no-plot
```

O estudo salva `burgers_shock_capture.npz` e, sem `--no-plot`,
`burgers_shock_capture.png` em `results/generated/`. O `.npz` também guarda
a viscosidade e a máscara de células ativas na última avaliação do RHS.

