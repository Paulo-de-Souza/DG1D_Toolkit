from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import math

# -----------------------------------------------------------------------------
# IMPORTAÇÕES DO NOSSO TOOLKIT
# -----------------------------------------------------------------------------
from dg1d_toolkit.dg1d_core import DGSpace1D
from dg1d_toolkit.operators import FluxProjection
from dg1d_toolkit.dg1d_integrators import RKSSP104_Step, RKSSP54_Step
from dg1d_toolkit.numericalfluxes import rusanov
from dg1d_toolkit.dg1d_limiters import SlopeLimiterN
from dg1d_toolkit.physicalfluxes import PhysicsFluxes


OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "generated"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# CONFIGURAÇÕES FÍSICAS (Artigo Lu et al. 2023 - Seção 5.1)
# -----------------------------------------------------------------------------
L = 800.0             # Comprimento do tubo (m)
a = 1000.0            # Celeridade da onda (m/s)
g = 9.81              # Gravidade (m/s^2)
H_res = 20.0          # Nível do reservatório a montante (m)
V0 = 0.15             # Velocidade inicial (m/s)

# -----------------------------------------------------------------------------
# CONFIGURAÇÕES NUMÉRICAS
# -----------------------------------------------------------------------------
K = 50                # Número de elementos
N = 1                 # Ordem polinomial (N=1 com limitador)
xmin, xmax = 0.0, L
tmin, tmax = 0.0, 15.0 

# 1. Defina o fator de segurança (Cr) que deseja testar no momento
Cr = 0.5              # Altere para 0.7, 0.5, 0.3 ou 0.1 conforme o artigo

# 2. Calcule o delta x
dx = L / K            # Nota: 800/50 resulta em 16m (seu comentário antigo dizia 5m)

# 3. Calcule o dt isolando a fórmula de estabilidade de Galerkin Descontínuo
dt = Cr * (dx / (a * (2 * N + 1)))

# 4. Calcule o número de passos necessários para atingir tmax
# Usamos math.ceil para arredondar para cima e garantir que a simulação chegue aos 15s
Nsteps = math.ceil(tmax / dt)

print(f"Para Cr={Cr}: dx={dx}m, dt={dt:.5f}s, Passos={Nsteps}")

# -----------------------------------------------------------------------------
# 1. AS FÍSICAS DO GOLPE DE ARIETE (Usando sua estrutura elegante)
# -----------------------------------------------------------------------------
def water_hammer_flux(H, V):
    """ Fluxos Conservativos do Golpe de Ariete """
    return [(a**2 / g) * V, g * H]

# Como a tubulação é lisa, não temos termo fonte de atrito (J = 0)!

# -----------------------------------------------------------------------------
# 2. O OPERADOR ESPACIAL Lh
# -----------------------------------------------------------------------------
def Lh_waterhammer_instant(U_modal, t, space):
    H_hat, V_hat = U_modal[0], U_modal[1]
    
    Ht = np.zeros((space.Nldof, space.K + 2))
    Vt = np.zeros((space.Nldof, space.K + 2))
    Ht[:, 1:-1] = H_hat
    Vt[:, 1:-1] = V_hat
    
    H_int_L = np.dot(space.Flkp1[0, :], Ht[:, 1])
    V_int_L = np.dot(space.Flkp1[0, :], Vt[:, 1])
    H_int_R = np.dot(space.Frk[0, :], Ht[:, -2])
    V_int_R = np.dot(space.Frk[0, :], Vt[:, -2])
    
    # CONDIÇÕES DE CONTORNO
    # Borda Esquerda: Reservatório
    Ht[0, 0] = -H_int_L + 2.0 * H_res  
    Vt[0, 0] = V_int_L                 
    
    # Borda Direita: Fechamento Instantâneo (V = 0.0 direto no t > 0)
    V_valve = 0.0 
    Ht[0, -1] = H_int_R                
    Vt[0, -1] = -V_int_R + 2.0 * V_valve 
    
    # PROJEÇÃO DE FLUXOS
    ft_list = FluxProjection(space, [Ht, Vt], water_hammer_flux)
    F_Ht, F_Vt = ft_list[0], ft_list[1]
    
    F_Ht[0, 0], F_Vt[0, 0] = water_hammer_flux(Ht[0, 0], Vt[0, 0])
    F_Ht[0, -1], F_Vt[0, -1] = water_hammer_flux(Ht[0, -1], Vt[0, -1])
    
    # FLUXOS NUMÉRICOS (Rusanov)
    C_local = np.ones(space.K + 1) * a
    flux_num_H = rusanov(space, Ht, F_Ht, C_local)
    flux_num_V = rusanov(space, Vt, F_Vt, C_local)
    
    # MONTAGEM (Sem termo fonte)
    rhs_H = space.InvM[:, None] * space.J[:]**(-1) * (np.dot(space.S.T, F_Ht[:, 1:-1]) + flux_num_H) 
    rhs_V = space.InvM[:, None] * space.J[:]**(-1) * (np.dot(space.S.T, F_Vt[:, 1:-1]) + flux_num_V)
    
    return [rhs_H, rhs_V]

# -----------------------------------------------------------------------------
# LOOP PRINCIPAL
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    space = DGSpace1D(K=K, N=N, xmin=xmin, xmax=xmax, quad_type='GL')
    
    # Instancia o Limitador que vai salvar a onda quadrada!
    limiter = SlopeLimiterN(space)
    
    # CI: Estado Permanente
    H_init_nodal = np.ones((space.nip, space.K)) * H_res
    V_init_nodal = np.ones((space.nip, space.K)) * V0
    
    invdiag = 1.0 / (np.sum(space.wi * space.psi.T**2, axis=1))
    U_modal = [
        (invdiag * np.dot(space.wi * space.psi.T, H_init_nodal).T).T,
        (invdiag * np.dot(space.wi * space.psi.T, V_init_nodal).T).T
    ]
    
    # Observador na válvula (x = L) para bater com os gráficos do artigo
    xi_coords = space.xc.flatten('F')
    idx_valve = np.argmin(np.abs(xi_coords - L))
    
    time_history = []
    H_valve_history = []
    
    t = tmin
    print("Iniciando Simulação (Fechamento Instantâneo)...")
    
    for n in range(1, Nsteps + 1):
        RHS_func = lambda U_list, time: Lh_waterhammer_instant(U_list, time, space)
        
        # Injetamos o limitador no integrador!
        U_modal = RKSSP54_Step(U_modal, t, dt, RHS_func, limiter)
        t += dt
        
        # Salva histórico a cada 0.05s para não explodir a RAM
        if n % int(0.05 / dt) == 0:
            H_nodal = np.dot(space.psi, U_modal[0]).flatten('F')
            time_history.append(t)
            H_valve_history.append(H_nodal[idx_valve])
            
            if t % 3.0 < dt:
                print(f"Tempo simulado: {t:.2f} s")

    np.savez(OUTPUT_DIR / "resultados_DG_WHv2.npz", 
                 tempo=time_history, 
                 H=H_valve_history)

    print(f"DG results saved in: {OUTPUT_DIR / 'resultados_DG_WHv2.npz'}")

    # Gráfico do Sensor na Válvula
    fig = plt.figure(figsize=(10, 5), dpi=100)
    plt.plot(time_history, H_valve_history, 'b-', lw=2, label='DG Toolkit (Limitado)')
    plt.title('Carga Piezométrica na Válvula (Fechamento Instantâneo)')
    plt.xlabel('Tempo (s)')
    plt.ylabel('H (m)')
    plt.grid(True)
    plt.legend(loc = 'upper right')
    plt.xlim([0, 15])
    plt.ylim([0, 50])
    fig.savefig(OUTPUT_DIR / "dg_instantaneous_valve_head.png", dpi=180, bbox_inches="tight")
    plt.show()
