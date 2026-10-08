from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# IMPORTAÇÕES DO NOSSO TOOLKIT
# -----------------------------------------------------------------------------
from dg1d_toolkit.dg1d_core import DGSpace1D
from dg1d_toolkit.operators import FluxProjection
from dg1d_toolkit.dg1d_integrators import RKSSP104_Step
from dg1d_toolkit.numericalfluxes import rusanov


OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "generated"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# CONFIGURAÇÕES FÍSICAS (Baseadas no Artigo - Seção 3.1)
# -----------------------------------------------------------------------------
L = 300.0             # Comprimento do tubo (m)
a = 1000.0            # Celeridade da onda (m/s)
D = 0.05              # Diâmetro (m)
f = 0.015             # Fator de atrito de Darcy-Weisbach
g = 9.81              # Gravidade (m/s^2)
H_res = 50.0          # Nível do reservatório a montante (m)
V0 = 0.412            # Velocidade inicial (m/s)
tc = 0.05             # Tempo de fechamento da válvula (s)

# Configurações Numéricas
K = 150               # Número de elementos
N = 2                 # Ordem polinomial
xmin, xmax = 0.0, L
tmin, tmax = 0.0, 6.0 # Simular 6 segundos
dt = 0.0002           # Delta t crítico devido a a=1000
Nsteps = int(tmax / dt)

# -----------------------------------------------------------------------------
# 1. AS FÍSICAS DO GOLPE DE ARIETE (Padrão PhysicsFluxes)
# -----------------------------------------------------------------------------
def water_hammer_flux(H, V):
    """ Fluxos Conservativos do Golpe de Ariete """
    # F_H = (a^2/g) * V
    # F_V = g * H
    return [(a**2 / g) * V, g * H]

def water_hammer_source(H, V):
    """ Termo fonte (Atrito constante - SFM) """
    S_H = np.zeros_like(H)
    S_V = - (f / (2 * D)) * V * np.abs(V)
    return [S_H, S_V]

# -----------------------------------------------------------------------------
# 2. O OPERADOR ESPACIAL Lh
# -----------------------------------------------------------------------------
def Lh_waterhammer(U_modal, t, space):
    H_hat, V_hat = U_modal[0], U_modal[1]
    
    # Prepara matrizes com padding (K+2) para o fluxo
    Ht = np.zeros((space.Nldof, space.K + 2))
    Vt = np.zeros((space.Nldof, space.K + 2))
    Ht[:, 1:-1] = H_hat
    Vt[:, 1:-1] = V_hat
    
    # Recupera valores internos nas bordas físicas da malha
    H_int_L = np.dot(space.Flkp1[0, :], Ht[:, 1])
    V_int_L = np.dot(space.Flkp1[0, :], Vt[:, 1])
    H_int_R = np.dot(space.Frk[0, :], Ht[:, -2])
    V_int_R = np.dot(space.Frk[0, :], Vt[:, -2])
    
    # -------------------------------------------------------------------------
    # CONDIÇÕES DE CONTORNO
    # -------------------------------------------------------------------------
    # Borda Esquerda (x=0): Reservatório Nível Constante
    Ht[0, 0] = -H_int_L + 2.0 * H_res  # Dirichlet em H
    Vt[0, 0] = V_int_L                 # Neumann/Extrapolação em V (Livre)
    
    # Borda Direita (x=L): Válvula Fechando
    V_valve = V0 * max(0.0, 1.0 - t/tc) # Fechamento linear em 0.05s
    Ht[0, -1] = H_int_R                # Neumann/Extrapolação em H (Livre)
    Vt[0, -1] = -V_int_R + 2.0 * V_valve # Dirichlet em V
    
    # -------------------------------------------------------------------------
    # PROJEÇÃO DOS FLUXOS FÍSICOS
    # -------------------------------------------------------------------------
    ft_list = FluxProjection(space, [Ht, Vt], water_hammer_flux)
    F_Ht, F_Vt = ft_list[0], ft_list[1]
    
    # Sobrescreve as bordas dos fluxos físicos usando as ghost cells
    F_Ht[0, 0], F_Vt[0, 0] = water_hammer_flux(Ht[0, 0], Vt[0, 0])
    F_Ht[0, -1], F_Vt[0, -1] = water_hammer_flux(Ht[0, -1], Vt[0, -1])
    
    # -------------------------------------------------------------------------
    # FLUXOS NUMÉRICOS E TERMO FONTE
    # -------------------------------------------------------------------------
    # Celeridade constante para definir o penalty do Rusanov
    C_local = np.ones(space.K + 1) * a
    
    flux_num_H = rusanov(space, Ht, F_Ht, C_local)
    flux_num_V = rusanov(space, Vt, F_Vt, C_local)
    
    # Projeção do Termo Fonte (Atrito) - Feito nos pontos de quadratura originais
    H_h = np.dot(space.psi, H_hat)
    V_h = np.dot(space.psi, V_hat)
    _, S_V_nodal = water_hammer_source(H_h, V_h)
    
    # Retorna o termo fonte nodal para modal
    S_V_modal = space.InvM[:, None] * np.dot(space.wi * space.psi.T, S_V_nodal)
    
    # -------------------------------------------------------------------------
    # MONTAGEM FINAL DO RHS
    # -------------------------------------------------------------------------
    rhs_H = space.InvM[:, None] * space.J[:]**(-1) * (np.dot(space.S.T, F_Ht[:, 1:-1]) + flux_num_H) 
    rhs_V = space.InvM[:, None] * space.J[:]**(-1) * (np.dot(space.S.T, F_Vt[:, 1:-1]) + flux_num_V) + S_V_modal
    
    return [rhs_H, rhs_V]

# -----------------------------------------------------------------------------
# LOOP PRINCIPAL E CAPTURA DE DADOS (Sensores Virtuais)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    space = DGSpace1D(K=K, N=N, xmin=xmin, xmax=xmax, quad_type='GL')
    
    # Condição Inicial: Estado Permanente
    H_init_nodal = np.ones((space.nip, space.K)) * H_res
    V_init_nodal = np.ones((space.nip, space.K)) * V0
    
    # Projeta CI para modal
    invdiag = 1.0 / (np.sum(space.wi * space.psi.T**2, axis=1))
    U_modal = [
        (invdiag * np.dot(space.wi * space.psi.T, H_init_nodal).T).T,
        (invdiag * np.dot(space.wi * space.psi.T, V_init_nodal).T).T
    ]
    
    # Sensores Virtuais como no artigo: x = 50m (Sensor 1) e x = 200m (Sensor 4)
    # Achamos o elemento K mais próximo do sensor e o nó interno correspondente
    xi_coords = space.xc.flatten('F')
    idx_50m = np.argmin(np.abs(xi_coords - 50.0))
    idx_200m = np.argmin(np.abs(xi_coords - 200.0))
    
    # Históricos para plotagem
    time_history = []
    H_50_history, H_200_history = [], []
    V_50_history, V_200_history = [], []
    
    t = tmin
    print("Iniciando Simulação do Golpe de Ariete...")
    
    # Salva o t=0
    H_nodal = np.dot(space.psi, U_modal[0]).flatten('F')
    V_nodal = np.dot(space.psi, U_modal[1]).flatten('F')
    time_history.append(t)
    H_50_history.append(H_nodal[idx_50m])
    H_200_history.append(H_nodal[idx_200m])
    V_50_history.append(V_nodal[idx_50m])
    V_200_history.append(V_nodal[idx_200m])

    # =========================================================================
    # TRECHO 1: INSERIR ANTES DO LOOP DE TEMPO (Linha ~105)
    # =========================================================================
    # H_full_history = [H_nodal.copy()]
    # V_full_history = [V_nodal.copy()]
    # time_full_history = [t]
    
    # Loop de Tempo
    for n in range(1, Nsteps + 1):
        RHS_func = lambda U_list, time: Lh_waterhammer(U_list, time, space)
        U_modal = RKSSP104_Step(U_modal, t, dt, RHS_func)
        t += dt
        
        # Amostrar dados a cada 0.01s (como os sensores do artigo) para os gráficos
        if n % int(0.01 / dt) == 0:
            H_nodal = np.dot(space.psi, U_modal[0]).flatten('F')
            V_nodal = np.dot(space.psi, U_modal[1]).flatten('F')
            
            time_history.append(t)
            H_50_history.append(H_nodal[idx_50m])
            H_200_history.append(H_nodal[idx_200m])
            V_50_history.append(V_nodal[idx_50m])
            V_200_history.append(V_nodal[idx_200m])

            # =================================================================
            # INSERIR DENTRO DO IF (Logo após salvar os sensores)
            # =================================================================
            # H_full_history.append(H_nodal.copy())
            # V_full_history.append(V_nodal.copy())
            # time_full_history.append(t)
            
            if t % 1.0 < dt:
                print(f"Tempo simulado: {t:.2f} s")

    # -------------------------------------------------------------------------
    # SALVAR RESULTADOS DO DG
    # -------------------------------------------------------------------------
    np.savez(OUTPUT_DIR / "resultados_DG.npz", 
             tempo=time_history, 
             H_50=H_50_history, H_200=H_200_history,
             V_50=V_50_history, V_200=V_200_history)
    print(f"DG results saved in: {OUTPUT_DIR / 'resultados_DG.npz'}")

    

    # -------------------------------------------------------------------------
    # GERAÇÃO DOS GRÁFICOS (Baseado nas Figuras 5b e 6 do Artigo)
    # -------------------------------------------------------------------------
    # Figura 5: Carga Piezométrica (H)
    fig_head = plt.figure(figsize=(10, 5), dpi=100)
    plt.plot(time_history, H_50_history, 'b-', label='Sensor x = 50 m')
    plt.plot(time_history, H_200_history, 'r--', label='Sensor x = 200 m')
    plt.title('Carga Piezométrica no Golpe de Ariete (Replicação Fig. 5)')
    plt.xlabel('Time (s)')
    plt.ylabel('Piezometric Head (m)')
    plt.grid(True)
    plt.legend()
    plt.xlim([0, 6])
    fig_head.savefig(OUTPUT_DIR / "dg_head_sensors.png", dpi=180, bbox_inches="tight")
    
    # # Figura 6: Velocidade (V)
    fig_velocity = plt.figure(figsize=(10, 5), dpi=100)
    plt.plot(time_history, V_50_history, 'b-', label='Sensor x = 50 m')
    plt.plot(time_history, V_200_history, 'r--', label='Sensor x = 200 m')
    plt.title('Velocidade de Escoamento (Replicação Fig. 6)')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.grid(True)
    plt.legend()
    plt.xlim([0, 6])
    fig_velocity.savefig(OUTPUT_DIR / "dg_velocity_sensors.png", dpi=180, bbox_inches="tight")
    
    plt.show()

    # from dg1d_toolkit.dg1d_postprocessing import create_gif
    # =========================================================================
    # TRECHO 2: GERAÇÃO DOS GIFS (Usando o dg1d_postprocessing)
    # =========================================================================
    # print("Gerando animações com o Toolkit...")
    
    # # Converte para arrays numpy
    # # (assumindo que o seu create_gif itera sobre o primeiro índice do array)
    # H_full_array = np.array(H_full_history)
    # V_full_array = np.array(V_full_history)
    # x_plot = space.xc.flatten('F')
    
    # # Define limites dinâmicos
    # ymin_H, ymax_H = np.min(H_full_array) - 5, np.max(H_full_array) + 5
    # ymin_V, ymax_V = np.min(V_full_array) - 0.2, np.max(V_full_array) + 0.2
    
    # -------------------------------------------------------------------------
    # GIF 1: Carga Piezométrica (H) usando sua função
    # -------------------------------------------------------------------------
    # create_gif(
    #     x=x_plot,
    #     u_history=H_full_array.T, 
    #     dt=0.01, # Passamos 0.01 porque salvamos os dados a cada 0.01s no loop
    #     folder="H_animation",
    #     filename="waterhammer_H.gif",
    #     title="Carga Piezométrica (H)",
    #     xlabel="Distância x (m)",
    #     ylabel="H (m)",
    #     xlim=(0.0, L),
    #     ylim=(ymin_H, ymax_H),
    #     duration=80,
    # )
    # print("GIF 1 (H) gerado com sucesso!")

    # # -------------------------------------------------------------------------
    # # GIF 2: Velocidade (V) usando sua função
    # # -------------------------------------------------------------------------
    # create_gif(
    #     x=x_plot,
    #     u_history=V_full_array.T,
    #     dt=0.01,
    #     folder="V_animation",
    #     filename="waterhammer_V.gif",
    #     title="Velocidade do Escoamento (V)",
    #     xlabel="Distância x (m)",
    #     ylabel="V (m/s)",
    #     xlim=(0.0, L),
    #     ylim=(ymin_V, ymax_V),
    #     duration=80,
    # )
    # print("GIF 2 (V) gerado com sucesso!")

    # -------------------------------------------------------------------------
    # GIF 3: Subplot Combinado (Lógica Customizada)
    # -------------------------------------------------------------------------
    # Como o seu create_gif atual plota 1 eixo por vez, usamos esse bloco 
    # para gerar a animação dupla.
    # import matplotlib.animation as animation
    # import os
    
    # os.makedirs("Combined_animation", exist_ok=True)
    # fig3, (ax3a, ax3b) = plt.subplots(2, 1, figsize=(8, 8), dpi=100, sharex=True)
    
    # line_H_comb, = ax3a.plot(x_plot, H_full_array[0], 'b-', lw=2)
    # ax3a.set_xlim(0, L)
    # ax3a.set_ylim(ymin_H, ymax_H)
    # ax3a.set_title("Golpe de Ariete: H e V combinados")
    # ax3a.set_ylabel("H (m)")
    # ax3a.grid(True)
    # time_text3 = ax3a.text(0.02, 0.90, '', transform=ax3a.transAxes, fontsize=12)
    
    # line_V_comb, = ax3b.plot(x_plot, V_full_array[0], 'r-', lw=2)
    # ax3b.set_ylim(ymin_V, ymax_V)
    # ax3b.set_xlabel("Distância x (m)")
    # ax3b.set_ylabel("V (m/s)")
    # ax3b.grid(True)
    
    # def update_gif3(frame):
    #     line_H_comb.set_ydata(H_full_array[frame])
    #     line_V_comb.set_ydata(V_full_array[frame])
    #     # Multiplicamos o frame por 0.01 pois esse é o nosso dt de salvamento
    #     time_text3.set_text(f'Tempo: {frame * 0.01:.2f} s') 
    #     return line_H_comb, line_V_comb, time_text3

    # ani3 = animation.FuncAnimation(fig3, update_gif3, frames=len(H_full_array), blit=True)
    # ani3.save("Combined_animation/waterhammer_Combinado.gif", writer='pillow', fps=1000//80)
    # plt.close(fig3)
    # print("GIF 3 (Combinado) gerado com sucesso!")
