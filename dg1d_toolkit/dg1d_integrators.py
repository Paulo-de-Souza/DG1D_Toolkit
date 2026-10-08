import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, List, Optional

def Euler_Step(U_modal: List[np.ndarray], t: float, dt: float, 
    Lh_operator: Callable, 
    limiter_func: Optional[Callable] = None) -> List[np.ndarray]:
    """
    Avança a solução em um passo de tempo usando o Método de Euler Explícito.
    
    Parâmetros:
    ----------
    U_modal : List[np.ndarray]
        Lista contendo os coeficientes modais (graus de liberdade) do estado atual.
        Pode conter 1 variável (ex: Burgers) ou N variáveis (ex: Equações de Euler).
    t : float
        Tempo atual da simulação.
    dt : float
        Tamanho do passo de tempo (calculado via CFL).
    Lh_operator : Callable
        Função que representa o Operador Espacial DG. 
        Deve possuir a assinatura: Lh(U_modal, t) -> List[np.ndarray]
    limiter_func : Callable, opcional
        Função para aplicar o Slope Limiter na solução. 
        Deve possuir a assinatura: limiter(variavel_modal) -> np.ndarray
        
    Retorna:
    -------
    U_new : List[np.ndarray]
        Lista com as variáveis de estado atualizadas no tempo t + dt.
    """
    # 1. Avalia o operador espacial (Derivada no tempo)
    dU = Lh_operator(U_modal, t)
    
    # 2. Avança no tempo para cada equação do sistema
    U_new = []
    for u, du in zip(U_modal, dU):
        u_next = u + dt * du
        U_new.append(u_next)
        
    # 3. Aplica Corretores Espaciais (Slope Limiters), se fornecidos
    if limiter_func is not None:
        for i in range(len(U_new)):
            U_new[i] = limiter_func(U_new[i])
            
    return U_new


def RK4_Step(
    U_modal: List[np.ndarray], 
    t: float, 
    dt: float, 
    Lh_operator: Callable, 
    limiter_func: Optional[Callable] = None
) -> List[np.ndarray]:
    """
    Avança a solução em um passo de tempo usando o Método RK4 Clássico.
    Agnóstico à física e capaz de lidar com N equações acopladas.
    """
    num_vars = len(U_modal)
    
    # Helper interno para aplicar o limitador a todas as variáveis da lista
    def apply_limiter(U_state):
        if limiter_func is not None:
            return [limiter_func(u) for u in U_state]
        return U_state

    # Estágio 1 (k1) - Sondagem inicial
    k1 = Lh_operator(U_modal, t)
    
    U1 = [U_modal[i] + 0.5 * dt * k1[i] for i in range(num_vars)]
    U1 = apply_limiter(U1)
    
    # Estágio 2 (k2) - Sondagem no ponto médio
    k2 = Lh_operator(U1, t + 0.5 * dt)
    
    U2 = [U_modal[i] + 0.5 * dt * k2[i] for i in range(num_vars)]
    U2 = apply_limiter(U2)
    
    # Estágio 3 (k3) - Nova sondagem no ponto médio
    k3 = Lh_operator(U2, t + 0.5 * dt)
    
    U3 = [U_modal[i] + dt * k3[i] for i in range(num_vars)]
    U3 = apply_limiter(U3)
    
    # Estágio 4 (k4) - Sondagem no final do passo
    k4 = Lh_operator(U3, t + dt)
    
    # Avanço no Tempo (Média Ponderada)
    U_new = [
        U_modal[i] + (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i])
        for i in range(num_vars)
    ]
    U_new = apply_limiter(U_new)
    
    return U_new

def RKSSP54_Step(
    U_modal: List[np.ndarray], 
    t: float, 
    dt: float, 
    Lh_operator: Callable, 
    limiter_func: Optional[Callable] = None
) -> List[np.ndarray]:
    """ Integrador RK SSP 5 Estágios / 4ª Ordem (Spiteri-Ruuth) """
    num_vars = len(U_modal)
    
    def apply_limiter(U_state):
        if limiter_func is not None:
            return [limiter_func(u) for u in U_state]
        return U_state

    # Estágio 1
    k1 = Lh_operator(U_modal, t)
    U1 = [U_modal[i] + 0.39175222700392 * dt * k1[i] for i in range(num_vars)]
    U1 = apply_limiter(U1)
    
    # Estágio 2
    k2 = Lh_operator(U1, t + 0.39175222700392 * dt)
    U2 = [0.44437049406734 * U_modal[i] + 0.55562950593266 * U1[i] + 0.36841059262959 * dt * k2[i] for i in range(num_vars)]
    U2 = apply_limiter(U2)
    
    # Estágio 3
    k3 = Lh_operator(U2, t + 0.58607968896780 * dt)
    U3 = [0.62010185138540 * U_modal[i] + 0.37989814861460 * U2[i] + 0.25189177424738 * dt * k3[i] for i in range(num_vars)]
    U3 = apply_limiter(U3)
    
    # Estágio 4
    k4 = Lh_operator(U3, t + 0.47454236302687 * dt)
    U4 = [0.17807995410773 * U_modal[i] + 0.82192004589227 * U3[i] + 0.54497475021237 * dt * k4[i] for i in range(num_vars)]
    U4 = apply_limiter(U4)
    
    # Estágio 5 (Passo Final)
    k5 = Lh_operator(U4, t + 0.93501063100924 * dt)
    U_new = [
        0.00683325884039 * U_modal[i] + 
        0.51723167208978 * U2[i] + 
        0.12759831133288 * U3[i] + 
        0.34833675773694 * U4[i] + 
        0.08460416338212 * dt * k4[i] + 
        0.22600748319395 * dt * k5[i] 
        for i in range(num_vars)
    ]
    U_new = apply_limiter(U_new)
    
    return U_new

def RKSSP104_Step(
    U_modal: List[np.ndarray], 
    t: float, 
    dt: float, 
    Lh_operator: Callable, 
    limiter_func: Optional[Callable] = None
) -> List[np.ndarray]:
    """ Integrador RK SSP 10 Estágios / 4ª Ordem """
    num_vars = len(U_modal)
    
    def apply_limiter(U_state):
        if limiter_func is not None:
            return [limiter_func(u) for u in U_state]
        return U_state

    U_curr = [u.copy() for u in U_modal]
    
    # Estágios de 1 a 4 (Padrão repetitivo)
    for _ in range(4):
        k = Lh_operator(U_curr, t)
        U_curr = [U_curr[i] + (dt / 6.0) * k[i] for i in range(num_vars)]
        U_curr = apply_limiter(U_curr)
        
    # Salvamos o U4 e k5 para o Passo 10 e Passo 5
    U4 = [u.copy() for u in U_curr]
    k5 = Lh_operator(U4, t)
    
    # Estágio 5
    U_curr = [
        (3.0 / 5.0) * U_modal[i] + (2.0 / 5.0) * U4[i] + (1.0 / 15.0) * dt * k5[i]
        for i in range(num_vars)
    ]
    U_curr = apply_limiter(U_curr)
    
    # Estágios 6 a 9 (Novo padrão repetitivo)
    for _ in range(4):
        k = Lh_operator(U_curr, t)
        U_curr = [U_curr[i] + (dt / 6.0) * k[i] for i in range(num_vars)]
        U_curr = apply_limiter(U_curr)
        
    # U_curr neste momento é o U9
    U9 = U_curr
    k10 = Lh_operator(U9, t)
    
    # Estágio 10 (Passo Final)
    U_new = [
        (1.0 / 25.0) * U_modal[i] + 
        (9.0 / 25.0) * U4[i] + 
        (3.0 / 5.0) * U9[i] + 
        (3.0 / 50.0) * dt * k5[i] + 
        (1.0 / 10.0) * dt * k10[i]
        for i in range(num_vars)
    ]
    U_new = apply_limiter(U_new)
    
    return U_new