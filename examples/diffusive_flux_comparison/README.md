# Comparação BR1, LDG e BR2

Este caso usa a equação do calor periódica

```text
u_t = nu * u_xx,       u(x,0) = sin(2*pi*x)
```

e compara:

- BR1, que já existia na toolkit;
- LDG, portado da formulação ativa dos scripts de captura;
- BR2, portado do levantamento de saltos do script de Allen–Cahn.

Todos usam o mesmo `DGSpace1D`, a mesma projeção modal e o `RK4_Step` já
existente. A solução exata é usada apenas para imprimir o erro máximo.

```bash
python examples/diffusive_flux_comparison/run.py
python examples/diffusive_flux_comparison/run.py --K 48 --N 4 --nu 0.001 --t-end 0.02
```

Os resultados ficam em `results/generated/`. O parâmetro `--eta` controla a
constante de estabilização BR2; `--no-plot` suprime a figura.

