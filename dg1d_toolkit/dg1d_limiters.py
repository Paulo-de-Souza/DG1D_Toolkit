import numpy as np

def minmod(v):   
    """ Função matemática pura. Não precisa de classe. """
    m = np.size(v, 0)
    mfunc = np.zeros(np.size(v, 1))
    s_1 = np.sum(np.sign(v), axis=0) / m
    
    ids = (np.flatnonzero(np.abs(s_1) == 1)).astype(int)
    if len(ids) != 0:
        mfunc[ids] = s_1[ids] * np.amin(np.abs(v[:, ids]), axis=0)                
    return mfunc


class HierarchicalLimiter:
    def __init__(self, space):
        """
        O Limitador precisa conhecer a malha e os polinômios para funcionar.
        Passamos o objeto DGSpace1D inteiro para ele no construtor.
        """
        self.space = space

    def __call__(self, u):
        """
        Método mágico __call__. Permite que a instância da classe seja chamada
        como se fosse uma função lá dentro do Runge-Kutta!
        """
        uhatavg = u.copy() 
        
        # Calcula médias das células
        uhatavg[1:self.space.Nldof, :] = 0.
        uhavg = np.dot(self.space.psi, uhatavg)
        v = uhavg[0, :]

        ulimit = u.copy()
        eps0 = 1.0e-8
        eps1 = 1.0e-15

        # Valores nas bordas dos elementos (GLL / GL)
        ue1 = np.dot(self.space.Flkp1[0, :], u)
        ue2 = np.dot(self.space.Frk[0, :], u)

        vk = v
        vkm1 = np.zeros(self.space.K)
        vkm1[0] = v[0]
        vkm1[1:-1] = v[0:-2]
        
        vkp1 = np.zeros(self.space.K)
        vkp1[0:-2] = v[1:-1]
        vkp1[-1] = v[-1]
        # vkp1[:-1] = v[1:]
        # vkp1[-1] = v[-1]

        # Reconstrução e detecção
        ve1 = vk - minmod(np.array([vk - ue1, vk - vkm1, vkp1 - vk]))
        ve2 = vk + minmod(np.array([ue2 - vk, vk - vkm1, vkp1 - vk]))

        ids1 = np.nonzero(np.abs(ve1 - ue1) > eps0)[0]
        ids2 = np.nonzero(np.abs(ve2 - ue2) > eps0)[0] 
        ids = np.union1d(ids1, ids2)

        # Se alguma célula precisar de limitação
        if len(ids) > 0:
            idr = ids.copy()      
            uk = u.copy()        
            vk_lim = u.copy()
            
            ukm1 = np.zeros((self.space.Nldof, self.space.K))
            ukm1[:, 0] = uk[:, 0]
            ukm1[:, 1:-1] = uk[:, 0:-2]
        
            ukp1 = np.zeros((self.space.Nldof, self.space.K))
            ukp1[:, 0:-2] = uk[:, 1:-1]
            ukp1[:, -1] = uk[:, -1]

            # Aplica o limitador hierárquico descendo a ordem do polinômio
            for n in range(self.space.N, 0, -1):
                if len(idr) == 0:
                    break # Se não houver mais células problemáticas, sai do loop
                    
                C = np.sqrt((2*n + 1) * (2*n + 3))
                
                term1 = C * uk[n, idr]
                term2 = ukp1[n-1, idr] - uk[n-1, idr]
                term3 = uk[n-1, idr] - ukm1[n-1, idr]
                
                vk_lim[n, idr] = (1.0 / C) * minmod(np.array([term1, term2, term3]))
                ulimit[n, idr] = vk_lim[n, idr]
                
                # Remove elementos que já estão suaves
                idslim = np.where(np.abs(vk_lim[n, idr] - uk[n, idr]) < eps1)[0]
                idr = np.delete(idr, idslim, 0)
                
        return ulimit

class SlopeLimiterN:
    def __init__(self, space):
        self.space = space

    def _slope_limit_lin(self, uhl, xl, vm1, v0, vp1):
        """ Aplica o limitador linear (antigo SlopeLimitLin) """
        ulimit = uhl.copy()
        h = xl[-1, :] - xl[0, :]
        x0 = np.ones((self.space.Nldof, 1)) * (xl[0, :] + h / 2.0)
        hN = np.ones((self.space.Nldof, 1)) * h
        
        # Usa a matriz de colocação pura que você salvou no space!
        ux = (2.0 / hN) * np.dot(self.space.Dhj, uhl)
        
        slope = np.zeros((3, len(vp1)))
        slope[0, :] = ux[0, :]
        slope[1, :] = (vp1 - v0) / h
        slope[2, :] = (v0 - vm1) / h
        
        ulimit = np.ones((self.space.Nldof, 1)) * v0 + (xl - x0) * (np.ones((self.space.Nldof, 1)) * minmod(slope))
        return ulimit

    def __call__(self, u):
        uhatavg = u.copy()
        uhatlin = u.copy()    
        
        uh = np.dot(self.space.psi, u)
        
        uhatavg[1:self.space.Nldof, :] = 0.0
        uhavg = np.dot(self.space.psi, uhatavg)
        v = uhavg[0, :]

        ulimit = uh.copy()
        eps0 = 1.0e-8

        ue1 = np.dot(self.space.Flkp1[0, :], u)
        ue2 = np.dot(self.space.Frk[0, :], u)

        vk = v
        vkm1 = np.zeros(self.space.K)
        vkm1[0] = v[0]
        vkm1[1:-1] = v[0:-2]
        
        vkp1 = np.zeros(self.space.K)
        vkp1[0:-2] = v[1:-1]
        vkp1[-1] = v[-1]

        ve1 = vk - minmod(np.array([vk - ue1, vk - vkm1, vkp1 - vk]))
        ve2 = vk + minmod(np.array([ue2 - vk, vk - vkm1, vkp1 - vk]))

        ids1 = np.nonzero(np.abs(ve1 - ue1) > eps0)[0]
        ids2 = np.nonzero(np.abs(ve2 - ue2) > eps0)[0] 
        ids = np.union1d(ids1, ids2)

        if len(ids) > 0:
            uhatlin[2:self.space.Nldof, :] = 0.0
            uhl = np.dot(self.space.psi, uhatlin)
            ulimit[:, ids] = self._slope_limit_lin(uhl[:, ids], self.space.xc[:, ids], vkm1[ids], vk[ids], vkp1[ids])
        
        uhatlim = np.dot(np.linalg.inv(self.space.psi), ulimit)
        return uhatlim