from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import sympy
from sympy.utilities.lambdify import lambdify
from dg1d_toolkit.numericalfluxes import lax_friedrichs, br1_flux
import math

# -----------------------------------------------------------------------------
# IMPORTAÇÕES DA NOSSA TOOLKIT VALIDADA
# -----------------------------------------------------------------------------
from dg1d_toolkit.dg1d_core import DGSpace1D
from dg1d_toolkit.operators import FluxProjection
from dg1d_toolkit.dg1d_integrators import RKSSP104_Step


OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "generated"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# CONFIGURAÇÕES E FÍSICA ESPECÍFICA (Burgers Viscoso)
# -----------------------------------------------------------------------------
# Parâmetros Base
K = 20          
N = 10          
eps = 0.07      
xmin, xmax = 0.0, 2.0 * np.pi 
tmin, tmax = 0.0, 0.5
dt = 0.00125    
Nsteps = 400  

# Funções da Física
def f_burgers(u):
    return (u**2) / 2.0

def max_speed_burgers(u):
    return np.max(np.abs(u))

# Criação da Solução Analítica (Para CI e CC)
x_sym, nu_sym, t_sym = sympy.symbols('x nu t')
phi_sym = sympy.exp(-(x_sym-4*t_sym)**2/(4*nu_sym*(t_sym+1))) + sympy.exp(-(x_sym-4*t_sym-2*np.pi)**2/(4*nu_sym*(t_sym+1)))
phiprime_sym = phi_sym.diff(x_sym)
ui_sym = -2 * nu_sym * (phiprime_sym/phi_sym) + 4
ufunc = lambdify((t_sym, x_sym, nu_sym), ui_sym)

# O CÓDIGO DO Lh (Usando a matemática raiz do orientador com o nosso tolkit)
def Lh_burgers_viscoso(U_modal, t, space):
    # 1. Recupera espaço nodal
    uh = np.dot(space.psi, U_modal)
    C = max_speed_burgers(uh)
    
    # 2. Aloca arrays (Exatamente como o original)
    ut = np.zeros((space.Nldof, space.K + 2, 2))
    qt = np.zeros((space.Nldof, space.K + 2))
    epst = np.zeros((space.nip, space.K + 2))
    
    ut[:, 1:-1, 0] = U_modal.copy()
    epst[:, :] = eps 
    
    # 3. Contorno de Dirichlet
    uhL = np.dot(space.Flkp1[0, :], ut[:, 1, 0])
    ut[0, 0, 0] = -uhL + 2.0 * (ufunc(t, 0.0, eps))
    ut[0, -1, 0] = ut[0, 0, 0] 

    # 4. CÁLCULO DE qhL e qhR (Contornos do BR1) USANDO O TOOLKIT
    # 4.1. Borda Esquerda (Passamos apenas a fatia dos índices 0, 1 e 2)
    fluxo_borda_L = br1_flux(space, ut[:, 0:3, 0], epst[:, 0:3])
    qhL = space.InvM[:, None] * space.J[0]**(-1) * (
        - np.sqrt(epst[0, 1:2]) * np.dot(space.S.T, ut[:, 1:2, 0]) # Volume do 1º elemento
        + fluxo_borda_L
    )
    qt[0, 0] = np.dot(space.Flkp1[0, :], qhL.flatten())

    # 4.2. Borda Direita (Passamos apenas a fatia dos últimos 3 índices: -3, -2 e -1)
    fluxo_borda_R = br1_flux(space, ut[:, -3:, 0], epst[:, -3:])
    qhR = space.InvM[:, None] * space.J[-1]**(-1) * (
        - np.sqrt(epst[0, -2:-1]) * np.dot(space.S.T, ut[:, -2:-1, 0]) # Volume do último elemento
        + fluxo_borda_R)
    
    qt[0, -1] = np.dot(space.Frk[0, :], qhR.flatten())

    # 5. Projeção de Fluxo Advectivo (Usamos a nossa já validada!)
    # Prepara o padding para a projeção
    ut_padded = np.zeros((space.Nldof, space.K + 2))
    ut_padded[:, :] = ut[:, :, 0]
    
    ft_list = FluxProjection(space, [ut_padded], f_burgers)
    ft = ft_list[0]
    
    # Preenche o contorno da projeção
    ft[0, 0] = f_burgers(ut[0, 0, 0])
    ft[0, -1] = ft[0, 0]

    # Passo 6: Atualização de Q (A variável auxiliar do BR1)
    fluxo_borda_q = br1_flux(space, ut[:,:,0], epst)
    
    qt[:, 1:-1] = space.InvM[:, None] * space.J[:]**(-1) * (
        - np.sqrt(epst[0, 1:-1]) * np.dot(space.S.T, ut[:, 1:-1, 0])  # Volume
        + fluxo_borda_q                                               # Bordas
    )

    # Passo 7: Atualização Final de U (Lax-Friedrichs Advectivo + BR1 Difusivo)
    fluxo_borda_adv = lax_friedrichs(space, ut[:,:,0], ft, C)
    fluxo_borda_dif = br1_flux(space, qt, epst)
    
    ut[:, 1:-1, 1] = space.InvM[:, None] * space.J[:]**(-1) * ( 
        np.dot(space.S.T, ft[:, 1:-1])                            # Volume Advectivo
        - np.sqrt(epst[0, 1:-1]) * np.dot(space.S.T, qt[:, 1:-1]) # Volume Difusivo
        + fluxo_borda_adv                                         # Bordas Advectivas
        + fluxo_borda_dif                                         # Bordas Difusivas
    )
    
    return ut[:, 1:-1, 1].copy()

# -----------------------------------------------------------------------------
# LOOP PRINCIPAL
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Inicializa Malha
    space = DGSpace1D(K=K, N=N, xmin=xmin, xmax=xmax, quad_type='GL')
    
    # Condição Inicial Analítica
    xi_coords = space.xc.flatten('F')
    uf_initial = np.asarray([ufunc(0.0, x0, eps) for x0 in xi_coords])
    uf_reshaped = uf_initial.reshape((space.K, space.nip)).T
    
    invdiag = 1.0 / (np.sum(space.wi * space.psi.T**2, axis=1))
    b = np.dot(space.wi * space.psi.T, uf_reshaped)
    U_modal = (invdiag * b.T).T
    
    # Históricos
    u_history = np.zeros((space.nip * space.K, Nsteps + 1))
    u_history[:, 0] = np.dot(space.psi, U_modal).flatten('F')

    # Criamos a matriz para o histórico analítico também
    u_analytical = np.zeros((space.nip * space.K, Nsteps + 1))
    u_analytical[:, 0] = uf_initial
    
    t = tmin
    print("Iniciando simulação viscosa...")
    for n in range(1, Nsteps + 1):
        RHS_func = lambda U_list, time: [Lh_burgers_viscoso(U_list[0], time, space)]
        
        U_resultado = RKSSP104_Step([U_modal], t, dt, RHS_func)
        U_modal = U_resultado[0]
        t += dt
        
        u_history[:, n] = np.dot(space.psi, U_modal).flatten('F')
        # Salva analítico no instante atual (t = n * dt)
        u_analytical[:, n] = np.asarray([ufunc(t, x0, eps) for x0 in xi_coords])
   

    # Pós-Processamento (Os múltiplos plots)
    plt.figure(2, figsize=(8,6), dpi=100)
    plt.clf()
    x = xi_coords
    
    # tstep = 1
    plt.plot(x, u_history[:, 1], ls='-', lw=2, color='tab:blue', label='Numérico')    
    plt.plot(x, u_analytical[:, 1], ls='--', lw=2, color='tab:red', label='Analítico')
    
    # tstep = 250
    plt.plot(x, u_history[:, 250], ls='-', lw=2, color='tab:blue') 
    plt.plot(x, u_analytical[:, 250], ls='--', lw=2, color='tab:red')
    
    # tstep = 400
    plt.plot(x, u_history[:, 400], ls='-', lw=2, color='tab:blue')
    plt.plot(x, u_analytical[:, 400], ls='--', lw=2, color='tab:red')
    
    plt.xlim([0, 2*np.pi])
    plt.ylim([0, 8])
    plt.legend()
    plt.title("Burgers Viscoso (DG1D Toolkit vs Analítica)")
    plt.grid(True)
    plt.savefig(OUTPUT_DIR / "burgers_viscous_comparison.png", dpi=180, bbox_inches="tight")
    plt.show()
