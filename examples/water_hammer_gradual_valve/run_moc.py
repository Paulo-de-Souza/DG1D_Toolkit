from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "generated"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# PARÂMETROS DO GOLPE DE ARIETE (Xu et al., 2025)
# -----------------------------------------------------------------------------
L = 300.0             # Comprimento do tubo (m)
a = 1000.0            # Celeridade da onda (m/s)
D = 0.05              # Diâmetro (m)
f = 0.015             # Fator de atrito de Darcy-Weisbach
g = 9.81              # Gravidade (m/s^2)
H_res = 50.0          # Nível do reservatório a montante (m)
V0 = 0.412            # Velocidade inicial (m/s)
tc = 0.05             # Tempo de fechamento da válvula (s)
tmax = 6.0            # Tempo total de simulação (s)

# -----------------------------------------------------------------------------
# MALHA DO MOC (Estritamente dx = a * dt)
# -----------------------------------------------------------------------------
N_reaches = 150       # Número de trechos
dx = L / N_reaches    # dx = 2.0 m
dt = dx / a           # dt = 0.002 s
N_nodes = N_reaches + 1
N_steps = int(tmax / dt)

# Constantes das equações características
B = a / g
R = f * dx / (2.0 * g * D)

# Malha espacial
x = np.linspace(0, L, N_nodes)

# -----------------------------------------------------------------------------
# CONDIÇÕES INICIAIS (SFM - Steady Friction Model)
# -----------------------------------------------------------------------------
H = np.ones(N_nodes) * H_res
V = np.ones(N_nodes) * V0

# Para ser rigoroso com a física, a carga inicial decai linearmente devido ao atrito
for i in range(N_nodes):
    H[i] = H_res - (i * dx) * f * (V0**2) / (2.0 * g * D)

# Arrays para o próximo passo de tempo
H_new = np.zeros(N_nodes)
V_new = np.zeros(N_nodes)

# -----------------------------------------------------------------------------
# HISTÓRICOS PARA COMPARAÇÃO (Sensores em x = 50m e x = 200m)
# -----------------------------------------------------------------------------
idx_50m = int(50.0 / dx)
idx_200m = int(200.0 / dx)

time_history = [0.0]
H_50_history, H_200_history = [H[idx_50m]], [H[idx_200m]]
V_50_history, V_200_history = [V[idx_50m]], [V[idx_200m]]

# -----------------------------------------------------------------------------
# LOOP PRINCIPAL DO MOC
# -----------------------------------------------------------------------------
print("Calculando Método das Características (MOC)...")
for n in range(1, N_steps + 1):
    current_time = n * dt
    
    # 1. Nós Internos (C+ e C-)
    # Usando vetorização do NumPy para ser ultra rápido
    CP = H[:-2] + B * V[:-2] - R * V[:-2] * np.abs(V[:-2])
    CM = H[2:]  - B * V[2:]  + R * V[2:]  * np.abs(V[2:])
    
    H_new[1:-1] = (CP + CM) / 2.0
    V_new[1:-1] = (CP - CM) / (2.0 * B)
    
    # 2. Contorno a Montante (x = 0): Reservatório Constante
    # Usa a característica C- vinda do nó 1
    CM_bound = H[1] - B * V[1] + R * V[1] * np.abs(V[1])
    H_new[0] = H_res
    V_new[0] = (H_new[0] - CM_bound) / B
    
    # 3. Contorno a Jusante (x = L): Válvula Fechando
    # Usa a característica C+ vinda do penúltimo nó
    CP_bound = H[-2] + B * V[-2] - R * V[-2] * np.abs(V[-2])
    V_new[-1] = V0 * max(0.0, 1.0 - current_time / tc)
    H_new[-1] = CP_bound - B * V_new[-1]
    
    # Atualiza o estado
    H[:] = H_new[:]
    V[:] = V_new[:]
    
    # Salva os dados no histórico
    time_history.append(current_time)
    H_50_history.append(H[idx_50m])
    H_200_history.append(H[idx_200m])
    V_50_history.append(V[idx_50m])
    V_200_history.append(V[idx_200m])

print("MOC Finalizado!")

# -------------------------------------------------------------------------
# SALVAR RESULTADOS DO MOC
# -------------------------------------------------------------------------
np.savez(OUTPUT_DIR / "resultados_MOC.npz", 
            tempo=time_history, 
            H_50=H_50_history, H_200=H_200_history,
            V_50=V_50_history, V_200=V_200_history)
print(f"MOC results saved in: {OUTPUT_DIR / 'resultados_MOC.npz'}")

# -----------------------------------------------------------------------------
# PLOTAGEM DOS RESULTADOS DO MOC
# -----------------------------------------------------------------------------
fig_head = plt.figure(figsize=(10, 5), dpi=100)
plt.plot(time_history, H_50_history, 'b-', label='MOC: Sensor x = 50 m')
plt.plot(time_history, H_200_history, 'r--', label='MOC: Sensor x = 200 m')
plt.title('Carga Piezométrica - Solução de Referência (MOC)')
plt.xlabel('Tempo (s)')
plt.ylabel('H (m)')
plt.grid(True)
plt.legend()
plt.xlim([0, 6])
fig_head.savefig(OUTPUT_DIR / "moc_head_sensors.png", dpi=180, bbox_inches="tight")

fig_velocity = plt.figure(figsize=(10, 5), dpi=100)
plt.plot(time_history, V_50_history, 'b-', label='MOC: Sensor x = 50 m')
plt.plot(time_history, V_200_history, 'r--', label='MOC: Sensor x = 200 m')
plt.title('Velocidade de Escoamento - Solucao de Referencia (MOC)')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.grid(True)
plt.legend()
plt.xlim([0, 6])
fig_velocity.savefig(OUTPUT_DIR / "moc_velocity_sensors.png", dpi=180, bbox_inches="tight")
plt.show()
