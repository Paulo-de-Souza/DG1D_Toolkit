import numpy as np
import matplotlib.pyplot as plt
import math

### POLINOMIOS DE JACOBI PARA ENCONTRAR DIFERENTES FAMILIAS
# alpha = 0 | beta = 0 --> Polinomios de Legendre
# alpha = -0.5 | beta = -0.5 --> Polinomios de Chebyshev normalizados
def JacobiP(r: float, m: int, alpha: float, beta: float) -> float:
    """
    Avalia o Polinômio de Jacobi de ordem m no ponto r com pesos alpha e beta.

    ### Parameters
      r (float): O ponto a ser avaliado.
      m (int): O grau do polinomio.
      alpha (float): Parâmetro associado a alguma família dos polinômios de Jacobi
      beta (float): Parâmetro associado a alguma família dos polinômios de Jacobi

    ### Returns
      float: Retorna o resultado do polinomio de Jacobi para o ponto r.

    """
    if m == 0:
        return 1.0

    pn0 = 1.0
    pn1 = 0.5 * (alpha - beta + (alpha + beta + 2.0) * r)

    if m == 1:
        return pn1

    pn2 = 0.0
    for n in range(1, m):
        a1n = 2.0 * (n + 1) * (n + alpha + beta + 1) * (2.0 * n + alpha + beta)
        a2n = (2.0 * n + alpha + beta + 1) * (alpha**2 - beta**2)
        a3n = (2.0 * n + alpha + beta) * (2.0 * n + alpha + beta + 1) * (2.0 * n + alpha + beta + 2)
        a4n = 2.0 * (n + alpha) * (n + beta) * (2.0 * n + alpha + beta + 2)

        pn2 = (1.0 / a1n) * ((a2n + a3n * r) * pn1 - a4n * pn0)
        pn0 = pn1
        pn1 = pn2

    return pn2

### DERIVADA DE ORDEM 1 DOS POLINOMIOS DE JACOBI
def DJacobiP(r: float, m: int, alpha: float, beta: float) -> float:
    """
    Primeira derivada do Polinômio de Jacobi de ordem m avaliado no ponto r.
    Utiliza math.gamma para suportar adequadamente os parâmetros fracionários
    (ex: Chebyshev, onde alpha e beta = -0.5).
    """
    if m == 0:
        return 0.0

    if np.isclose(r, -1.0):
        c1 = ((-1.0)**(m-1))*0.5*(alpha+beta+m+1)
        return c1 * (
            math.gamma(m+beta+1) /
            (math.gamma(beta+2)*math.gamma(m))
        )

    if np.isclose(r, 1.0):
        c1 = 0.5*(alpha+beta+m+1)
        return c1 * (
            math.gamma(m+alpha+1) /
            (math.gamma(alpha+2)*math.gamma(m))
        )

    # Para pontos internos, calculamos o polinômio e usamos a relação para a derivada
    Pm  = JacobiP(r, m, alpha, beta)
    Pm1 = JacobiP(r, m-1, alpha, beta)

    b1n = (2*m + alpha + beta)*(1-r**2)
    b2n = m*(alpha-beta-(2*m+alpha+beta)*r)
    b3n = 2*(m+alpha)*(m+beta)

    return (b2n*Pm + b3n*Pm1)/b1n

def jacobi_roots(m: int, alpha: float, beta: float) -> np.ndarray:
    """
    Encontra as raízes do Polinômio de Jacobi de ordem m
    usando Newton-Raphson acoplado com Deflação Polinomial.

    ### Args:
      m (int): O grau do polinomio de Jacobi.
      alpha (float): Parâmetro associado a alguma família dos polinômios de Jacobi
      beta (float): Parâmetro associado a alguma família dos polinômios de Jacobi

    ### Returns:
      array: As m-raízes do Polinomio de Jacobi de grau m
    """
    if m == 0:
        return np.array([])

    x = np.zeros(m)
    epsilon = 1.0e-16

    for k in range(m):
        # Chute inicial: raízes do polinômio de Chebyshev
        r = -np.cos(np.pi * (2.0 * k + 1.0) / (2.0 * m))
        if k > 0:
            r = 0.5 * (r + x[k - 1])

        delta = 1.0
        while abs(delta) > epsilon:
            s = 0.0
            for i in range(k):
                s += 1.0 / (r - x[i])

            pm = JacobiP(r, m, alpha, beta)
            dpm = DJacobiP(r, m, alpha, beta)

            # Relação de Newton-Raphson com Deflação
            delta = -pm / (dpm - pm * s)
            r += delta

        x[k] = r

    return x

def jacobi_gauss_weights(x: np.ndarray, alpha: float, beta: float, n_points: int, quad_type: str = 'GL') -> np.ndarray:
    """
    Calcula os pesos correspondentes para as raízes da Quadratura
    do tipo Gauss-Legendre (GL) ou Gauss-Lobatto-Legendre (GLL) baseando-se no número de pontos.

    ### Args:
      x (np.array): Vetor de pontos a serem avaliados (zeros da quadratura).
      alpha (float): Parâmetro associado a alguma família dos polinômios de Jacobi
      beta (float): Parâmetro associado a alguma família dos polinômios de Jacobi
      n_points (int): Número de pontos para a quadratura ser exata
    ### Returns:
      array: Os pesos da quadratura
    """
    w = np.zeros(n_points)

    if quad_type == 'GL':
        c1 = (2.0**(alpha + beta + 1.0)) * math.gamma(alpha + n_points + 1.0) * math.gamma(beta + n_points + 1.0)
        c2 = math.gamma(n_points + 1.0) * math.gamma(alpha + beta + n_points + 1.0)
        coef = c1 / c2

        for i in range(n_points):
            dpm = DJacobiP(x[i], n_points, alpha, beta)
            w[i] = coef / ((1.0 - x[i]**2) * (dpm**2))

    elif quad_type == 'GLL':
        c1 = (2.0**(alpha + beta + 1.0)) * math.gamma(alpha + n_points) * math.gamma(beta + n_points)
        c2 = (n_points - 1.0)* math.gamma(n_points) * math.gamma(alpha + beta + n_points + 1.0)
        coef = c1 / c2

        for i in range(n_points):
            pm_minus_1 = JacobiP(x[i], n_points - 1, alpha, beta)
            c_i = coef / (pm_minus_1**2)

            if i == 0:
                w[i] = (beta + 1.0) * c_i
            elif i == n_points - 1:
                w[i] = (alpha + 1.0) * c_i
            else:
                w[i] = c_i
    else:
        raise ValueError("Integração inválida. Escolha 'GL' ou 'GLL'.")

    return w

def jacobi_gauss_quad(P: int, alpha: float = 0.0, beta: float = 0.0, quad_type: str = 'GL'):
    """
    Retorna as matrizes de raízes (xi) e pesos (wi) garantindo exatidão
    para a integração de um polinômio de grau P.

    Args:
      P (int): O grau do polinomio a ser integrado.
      alpha (float): Parâmetro associado a alguma família dos polinômios de Jacobi
      beta (float): Parâmetro associado a alguma família dos polinômios de Jacobi
      quad_type (str): Tipo de quadratura que deseja utilizar (por padrão utiliza-se Gauss-Legendre (GL))
    """
    if quad_type == 'GL':
        # GL integra exatamente até grau 2n - 1
        # P <= 2n - 1  -->  n >= (P + 1) / 2
        n_points = max(1, math.ceil((P + 1.0) / 2.0))
        # Para xi_GL, o grau do polinomio que da as raizes coincide com o numero de pontos
        xi = jacobi_roots(n_points, alpha, beta)
        wi = jacobi_gauss_weights(xi, alpha, beta, n_points, quad_type='GL')

    elif quad_type == 'GLL':
        # GLL integra exatamente até grau 2n - 3
        # P <= 2n - 3  -->  n >= (P + 3) / 2
        n_points = max(2, math.ceil((P + 3.0) / 2.0))

        # Quantidade de pontos para a quadratura GLL e definicao dos extremos
        xi = np.zeros(n_points)
        xi[0] = -1.0
        xi[-1] = 1.0

        if n_points > 2:
            # Para xi_GLL internos, o grau do polinomio que da as raizes se torna
            # n_pontos - 2, visto que os extremos sao pre-definidos
            x_internos = jacobi_roots(n_points - 2, alpha + 1.0, beta + 1.0)
            xi[1:-1] = x_internos

        wi = jacobi_gauss_weights(xi, alpha, beta, n_points, quad_type='GLL')

    else:
        raise ValueError("Integração inválida. Escolha 'GL' ou 'GLL'.")

    return xi, wi

import numpy as np
import math

def dPQ(ri: float, Q: int, alpha: float, beta: float, quad_type: str = 'GL') -> float:
    """
    Primeira derivada do polinômio auxiliar pi_Q(x).
    """
    if quad_type == 'GL':
        return DJacobiP(ri, Q, alpha, beta)

    elif quad_type == 'GLL':
        # Uso de np.isclose para comparação segura de ponto flutuante
        if np.isclose(ri, -1.0):
            c = 2.0 * (-1)**Q
            # math.factorial(n) substituído por math.gamma(n+1) para generalização
            return (c * math.gamma(Q + beta)) / (math.gamma(Q - 1) * math.gamma(beta + 2.0))

        elif np.isclose(ri, 1.0):
            return (-2.0 * math.gamma(Q + alpha)) / (math.gamma(Q - 1) * math.gamma(alpha + 2.0))

        else:
            return -2.0 * (Q - 1) * JacobiP(ri, Q - 1, alpha, beta)

    else:
        raise ValueError("Integração inválida. Escolha 'GL' ou 'GLL'.")
    

def d2PQ(ri: float, Q: int, alpha: float, beta: float, quad_type: str = 'GL') -> float:
    """
    Segunda derivada do polinômio auxiliar pi_Q(x).
    """
    if quad_type == 'GL':
        dp_jacobi = DJacobiP(ri, Q, alpha, beta)
        return ((alpha - beta + (alpha + beta + 2.0) * ri) / (1.0 - ri**2)) * dp_jacobi

    elif quad_type == 'GLL':
        if np.isclose(ri, -1.0):
            c = (((-1)**Q) * 2.0 * (alpha - (Q - 1) * (Q + alpha + beta))) / (beta + 2.0)
            return (c * math.gamma(Q + beta)) / (math.gamma(Q - 1) * math.gamma(beta + 2.0))

        elif np.isclose(ri, 1.0):
            c = (2.0 * (beta - (Q - 1) * (Q + alpha + beta))) / (alpha + 2.0)
            return (c * math.gamma(Q + alpha)) / (math.gamma(Q - 1) * math.gamma(alpha + 2.0))

        else:
            p_jacobi = JacobiP(ri, Q - 1, alpha, beta)
            dp_jacobi = (-2.0 * (Q - 1)) / ((1.0 - ri) * (1.0 + ri)) * p_jacobi
            return (alpha - beta + (alpha + beta) * ri) * dp_jacobi

    else:
        raise ValueError("Integração inválida. Escolha 'GL' ou 'GLL'.")


def DMatrix1D(xi: np.ndarray, nip: int, quad_type: str = 'GL', alpha: float = 0.0, beta: float = 0.0) -> np.ndarray:
    """
    Matriz de Diferenciação por Colocação.

    Args:
        xi (array) : pontos a serem avaliados
        nip (int) : grau do polinomio + 1 OU tamanho do vetor xi
        quad_type (str) : tipo de quadratura a ser usada (por padrão GL)
        alpha (float): Parâmetro associado a alguma família dos polinômios de Jacobi
        beta (float): Parâmetro associado a alguma família dos polinômios de Jacobi
    """
    if nip <= 1:
        return np.zeros((1, 1))

    D = np.zeros((nip, nip))

    # PRÉ-COMPUTAÇÃO: Avalia pi' e pi'' uma única vez para todos os pontos nodais
    pi_prime = np.array([dPQ(r, nip, alpha, beta, quad_type) for r in xi])
    pi_double_prime = np.array([d2PQ(r, nip, alpha, beta, quad_type) for r in xi])

    for i in range(nip):
        for j in range(nip):
            if i == j:
                # Diagonal principal
                D[i, i] = pi_double_prime[i] / (2.0 * pi_prime[i])
            else:
                # Elementos fora da diagonal
                D[i, j] = (pi_prime[i] / pi_prime[j]) * (1.0 / (xi[i] - xi[j]))

    return D