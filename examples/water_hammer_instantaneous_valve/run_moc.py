from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "generated"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# PARÂMETROS FÍSICOS (Lu et al. 2023 - Seção 5.1)
# -----------------------------------------------------------------------------
L = 800.0             # Comprimento do tubo (m)
a = 1000.0            # Celeridade da onda (m/s)
g = 9.81              # Gravidade (m/s^2)
H_res = 20.0          # Nível do reservatório a montante (m)
V0 = 0.15             # Velocidade inicial (m/s)
tmax = 15.0           # Tempo total de simulação (s)

# -----------------------------------------------------------------------------
# MALHA DO MOC (Estritamente dx = a * dt)
# -----------------------------------------------------------------------------
N_reaches = 32       # Número de trechos (dx = 5.0 m)
dx = L / N_reaches
dt = dx / a           # dt = 0.005 s
N_nodes = N_reaches + 1
N_steps = int(tmax / dt)

# Constante de impedância (sem atrito para este caso)
B = a / g

# Condição Inicial (Tubo liso = sem perda de carga inicial)
H = np.ones(N_nodes) * H_res
V = np.ones(N_nodes) * V0

H_new = np.zeros(N_nodes)
V_new = np.zeros(N_nodes)

# Históricos para plotagem (Sensor na Válvula)
time_history = [0.0]
H_valve_history = [H[-1]]

# -----------------------------------------------------------------------------
# LOOP PRINCIPAL DO MOC
# -----------------------------------------------------------------------------
print(f"Calculando Método das Características (dt = {dt} s)...")
for n in range(1, N_steps + 1):
    current_time = n * dt
    
    # 1. Nós Internos (C+ e C- puramente elásticos, sem atrito)
    CP = H[:-2] + B * V[:-2]
    CM = H[2:]  - B * V[2:]
    
    H_new[1:-1] = (CP + CM) / 2.0
    V_new[1:-1] = (CP - CM) / (2.0 * B)
    
    # 2. Contorno a Montante (x = 0): Reservatório Constante
    CM_bound = H[1] - B * V[1]
    H_new[0] = H_res
    V_new[0] = (H_new[0] - CM_bound) / B
    
    # 3. Contorno a Jusante (x = L): Fechamento Instantâneo
    CP_bound = H[-2] + B * V[-2]
    V_new[-1] = 0.0  # A válvula fecha no t=0 e fica fechada (V = 0 direto!)
    H_new[-1] = CP_bound - B * V_new[-1]
    
    # Atualiza o estado
    H[:] = H_new[:]
    V[:] = V_new[:]
    
    # Salva os dados no histórico da válvula
    time_history.append(current_time)
    H_valve_history.append(H[-1])

print("MOC Finalizado!")

# -----------------------------------------------------------------------------
# PLOTAGEM DO RESULTADO DO MOC
# -----------------------------------------------------------------------------
fig = plt.figure(figsize=(10, 5), dpi=100)
plt.plot(time_history, H_valve_history, 'k-', lw=1.5, label='MOC (Referência exata, Cr=1)')
plt.title('Carga Piezométrica na Válvula - Solução MOC')
plt.xlabel('Tempo (s)')
plt.ylabel('H (m)')
plt.grid(True)
plt.legend()
plt.xlim([0, 15])
plt.ylim([-5, 45])
fig.savefig(OUTPUT_DIR / "moc_instantaneous_valve_head.png", dpi=180, bbox_inches="tight")
plt.show()

np.savez(
    OUTPUT_DIR / "resultados_MOC_instantaneous_valve.npz",
    time=time_history,
    head_at_valve=H_valve_history,
)
