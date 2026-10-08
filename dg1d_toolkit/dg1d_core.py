import numpy as np
import math
from dg1d_toolkit.mathDG1D import JacobiP, jacobi_gauss_quad, DMatrix1D

class DGSpace1D:
    def __init__(self, K: int, N: int, xmin: float = 0.0, xmax: float = 1.0, 
                 quad_type: str = 'GL', P: int = None, nlP: int = None):
        # 1. Parâmetros Base
        self.K = K
        self.N = N
        self.Nldof = N + 1
        self.quad_type = quad_type
        self.xmin = xmin
        self.xmax = xmax

        # Grau para integracao linear
        self.P = P if P is not None else 2 * self.N

        # Grau para integracao nao-linear
        self.nlP = nlP if nlP is not None else 3 * self.N
        
        # 2. Geração da Malha
        self.Nv, self.VX, self.EToV = self._MeshGen1D()
        
        # 3. Quadratura Padrão (xi) e Matrizes associadas
        self.xi, self.wi = jacobi_gauss_quad(self.P, quad_type=self.quad_type)
        self.nip = len(self.xi)
        self.psi = self._LegendreBasis(self.xi)
        
        # 4. Quadratura de Superintegração (nlxi) e Base
        self.nlxi, self.nlwi = jacobi_gauss_quad(self.nlP, quad_type=self.quad_type)
        self.nlnip = len(self.nlxi)
        self.nlpsi = self._LegendreBasis(self.nlxi)
        
        # 5. Mapeamento Físico (Jacobiano)
        self.xc, self.J = self._Jacobian()
        
        # 6. Matrizes de Derivação, Rigidez, Massa e Liftings
        Dhj = DMatrix1D(self.xi, self.nip, quad_type=self.quad_type)
        self.Dhj = Dhj
        self.Dpsi = np.dot(Dhj, self.psi)
        
        self.S = self._StiffMatrix()
        self.diagM, self.InvM = self._MassMatrix(method='analytic')
        self.Flk, self.Frk, self.Flkp1, self.Frkm1 = self._LiftMatrix()

    def _MeshGen1D(self):
        """ Gerador de uma malha simples equidistante com K elementos. """
        Nv = self.K + 1
        VX = np.linspace(self.xmin, self.xmax, Nv)

        EToV = np.zeros((self.K, 2), dtype=int) 
        for k in range(self.K): 
            EToV[k, 0] = k 
            EToV[k, 1] = k + 1

        return Nv, VX, EToV

    def _Jacobian(self):
        """ Mapeia os pontos de quadratura do elemento de referência [-1,1] para os físicos. """
        J = 0.5 * (self.VX[1:] - self.VX[:-1])
        B = 0.5 * (self.VX[1:] + self.VX[:-1])
        
        xcoord = self.xi[:, None] * J[None, :] + B[None, :]
        return xcoord, J

    def _LegendreBasis(self, pontos_xi):
        """ Constrói a matriz de avaliação da base ortogonal de Legendre. """
        npts = len(pontos_xi)
        Lj = np.zeros((npts, self.Nldof)) 
        
        for j in range(self.Nldof):
            for i in range(npts):
                Lj[i, j] = JacobiP(pontos_xi[i], j, 0, 0)
                
        return Lj

    def _MassMatrix(self, method='analytic'):
        """ Constrói a diagonal da Matriz de Massa e sua respectiva Inversa. """
        diagM = np.zeros(self.Nldof)
        invDiagM = np.zeros(self.Nldof)
        
        if method == 'analytic':
            for n in range(self.Nldof):
                diagM[n] = 2.0 / (2.0 * n + 1.0)
                invDiagM[n] = 1.0 / diagM[n]
                
        elif method == 'quadrature':
            for i in range(self.Nldof):
                diagM[i] = np.sum(self.wi * (self.psi[:, i]**2))
                invDiagM[i] = 1.0 / diagM[i]
        else:
            raise ValueError("Erro: Método desconhecido. Escolha 'analytic' ou 'quadrature'.")
            
        return diagM, invDiagM

    def _StiffMatrix(self):
        """ Matriz de Rigidez (Advecção) """
        return np.dot(self.wi * self.psi.T, self.Dpsi)

    def _LiftMatrix(self):
        """ Constrói as Matrizes de Elevação (Lift Matrices). """
        Frk = np.ones((self.Nldof, self.Nldof))
        Flk = np.ones((self.Nldof, self.Nldof))
        Flkp1 = np.ones((self.Nldof, self.Nldof))
        Frkm1 = np.ones((self.Nldof, self.Nldof))
        
        Flkp1[:, 1:self.Nldof:2] = -1.0 * Flkp1[:, 1:self.Nldof:2]
        Frkm1[1:self.Nldof:2, :] = -1.0 * Frkm1[1:self.Nldof:2, :]
        
        Flk[:self.Nldof:2, 1:self.Nldof:2] = -1.0 * Flk[:self.Nldof:2, 1:self.Nldof:2]
        Flk[1:self.Nldof:2, :self.Nldof:2] = -1.0 * Flk[1:self.Nldof:2, :self.Nldof:2]
        
        return Flk, Frk, Flkp1, Frkm1