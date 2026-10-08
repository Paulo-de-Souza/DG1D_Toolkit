from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# =================================================================
# 1. IMPORTANDO AS FERRAMENTAS DO TOOLKIT
# =================================================================
from dg1d_toolkit.dg1d_core import DGSpace1D
from dg1d_toolkit.operators import FluxProjection
from dg1d_toolkit.dg1d_limiters import HierarchicalLimiter
from dg1d_toolkit.dg1d_integrators import RKSSP104_Step, RKSSP54_Step
from dg1d_toolkit.dg1d_postprocessing import create_gif
from dg1d_toolkit.numericalfluxes import lax_friedrichs


OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "generated"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =================================================================
# 2. DEFINIÇÃO DA FÍSICA (Burgers Inviscido)
# =================================================================
def f_burgers(u):
    return (u**2) / 2.0

def max_speed_burgers(u):
    return np.max(np.abs(u))

def condicao_inicial(x):
    """ Centraliza a fórmula da IC """
    return 0.5 + np.sin(2.0 * np.pi * x)

def contorno_dirichlet(uL, t):
    """ Centraliza a fórmula da BC para ser chamada no Lh """
    return -uL + 2.0 * (0.5 + np.sin(-2.0 * np.pi * uL * t))

# =================================================================
# 3. A MONTAGEM DO Lh PELO USUÁRIO 
# =================================================================
def Lh_burgers_otimizado(U_modal, t, space):
    # a) Criação das Células Fantasmas MODAIS
    U_modal_padded = np.zeros((space.N + 1, space.K + 2))
    U_modal_padded[:, 1:-1] = U_modal

    # b) Extrai a fronteira esquerda real usando a matriz de Lift
    uL = np.dot(space.Flkp1[0, :], U_modal_padded[:, 1])

    # c) Aplicação da Condição de Contorno no coeficiente modal médio (índice 0)
    U_modal_padded[0, 0] = contorno_dirichlet(uL, t)
    U_modal_padded[0, -1] = U_modal_padded[0, 0]

    # d) Projeção do Fluxo 
    ft_list = FluxProjection(space, [U_modal_padded], f_burgers)
    ft = ft_list[0] 

    # e) --- A CORREÇÃO DA GHOST CELL ---
    # Injetamos o fluxo físico avaliado nos coeficientes que acabamos de preencher
    ft[0, 0] = f_burgers(U_modal_padded[0, 0])
    ft[0, -1] = f_burgers(U_modal_padded[0, -1])

    # f) Velocidade Máxima Global 
    # Aqui sim vamos para o espaço nodal rapidinho só para achar o max_speed real
    uh = np.dot(space.psi, U_modal)
    max_speed = max_speed_burgers(uh)

    # g) FORMULAÇÃO MATRICIAL OTIMIZADA DO LAX-FRIEDRICHS (Operando em Modal!)
    ut = U_modal_padded # Renomeando apenas para a fórmula ficar igual à antiga
    
    RHS_vol = np.dot(space.S.T, ft[:, 1:-1])
    
    # RHS_num = (
    #     - 0.5 * np.dot(space.Flkp1, (ft[:, 2:] - max_speed * ut[:, 2:]))
    #     - 0.5 * np.dot(space.Frk,   (ft[:, 1:-1] + max_speed * ut[:, 1:-1]))
    #     + 0.5 * np.dot(space.Frkm1, (ft[:, :-2] + max_speed * ut[:, :-2]))
    #     + 0.5 * np.dot(space.Flk,   (ft[:, 1:-1] - max_speed * ut[:, 1:-1]))
    # )
    flux_num = lax_friedrichs(space,ut,ft,max_speed)
    # Soma tudo, multiplica pela inversa da Massa e pelo Jacobiano de uma vez
    RHS_final = space.InvM[:, None] * (space.J**(-1) * (RHS_vol + flux_num))
    
    return RHS_final

# =================================================================
# 4. CONFIGURAÇÃO DA SIMULAÇÃO E LOOP DE TEMPO
# =================================================================
if __name__ == "__main__":
    # Configuração Exata do Exemplo
    K = 250
    N = 4
    xmin, xmax = 0.0, 1.0
    tmin, tmax = 0.0, 0.5
    Nsteps = 62
    dt = 0.008

    # Instancia o Palco
    space = DGSpace1D(K=K, N=N, xmin=xmin, xmax=xmax, quad_type='GL')

    # Condição Inicial (Onda senoidal no domínio de 0 a 1)
    uic = condicao_inicial(space.xc)
    b = np.dot(space.wi * space.psi.T, uic)
    U_modal = space.InvM[:, None] * b

    # Inicializa o Histórico Temporal
    u_history = np.zeros((space.nip * space.K, Nsteps + 1))
    u_history[:, 0] = np.dot(space.psi, U_modal).T.flatten()

    # O limitador nasce recebendo o espaço
    meu_limitador = HierarchicalLimiter(space)

    # Laço de Tempo
    t = tmin
    print("Iniciando simulação...")
    for n in range(1, Nsteps + 1):
        # Tradutores Lambda (Convertem a linguagem de listas do integrador para a sua matriz)
        RHS_func = lambda U_list, time: [Lh_burgers_otimizado(U_list[0], time, space)]
        
        # Passamos a condição inicial DENTRO de uma lista: [U_modal]
        U_resultado_lista = RKSSP104_Step([U_modal], t, dt, RHS_func, meu_limitador)
        
        # Tiramos a matriz de dentro da lista para o próximo passo
        U_modal = U_resultado_lista[0]
        
        t += dt
        
        # Salva histórico nodal no instante atual
        uh = np.dot(space.psi, U_modal).T.flatten()
        u_history[:, n] = uh

    # =================================================================
    # 5. PÓS-PROCESSAMENTO (GIF)
    # =================================================================
    print("Simulação concluída! Gerando GIF...")
    x = space.xc.T.flatten()

    create_gif(
        x=x,
        u_history=u_history,
        dt=dt,
        folder=OUTPUT_DIR / "animation",
        filename="simulation.gif",
        title="Inviscid Burgers with Slope Limiter Tk",
        xlabel="x",
        ylabel="u",
        xlim=(xmin, xmax),
        ylim=(-0.8, 1.8),
        duration=80,
        plot_kwargs={
            "color": "tab:blue",
            "lw": 2,
        },
    )
