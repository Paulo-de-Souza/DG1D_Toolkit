import numpy as np
import math

class PhysicsFluxes:
    """
    Biblioteca de fluxos físicos para o solver espectral.
    Fornece as avaliações de f(uh) e termos fonte S(uh) nos pontos de integração.
    """

    @staticmethod
    def Burgers(uh, **kwargs):
        """
        Fluxo convectivo da equação de Burgers.

        Equação
        --------
        ∂u/∂t + ∂f(u)/∂x = 0

        com

            f(u) = u²/2

        Parameters
        ----------
        uh : ndarray
            Solução avaliada nos graus de liberdade.

        Returns
        -------
        ndarray
            Fluxo f(u).
        """
        return 0.5 * uh**2

    @staticmethod
    def AllenCahn(uh, ac_type='Burgers', **kwargs):
        """
        Parte convectiva da equação de Allen-Cahn.

        Dependendo do parâmetro ``ac_type`` são utilizados diferentes fluxos:

        - 'Burgers' :
            f(u) = u²/2

        - 'Cubic' :
            f(u) = u³ - u

        - 'PureDiff' :
            f(u) = 0

        - outro valor :
            f(u) = -βu

        Parameters
        ----------
        uh : ndarray
            Campo escalar.

        ac_type : str
            Tipo de fluxo utilizado.

        Returns
        -------
        ndarray
            Fluxo correspondente ao modelo escolhido.
        """
        if ac_type == 'Burgers':
            return (uh**2) / 2.0
        elif ac_type == 'Cubic':
            return uh**3 - uh
        elif ac_type == 'PureDiff':
            return np.zeros_like(uh)
        else:
            beta_fe = 0.5
            return -beta_fe * uh

    @staticmethod
    def ShallowWater(hh, uh, gravity=9.81, **kwargs):
        hf = hh * uh
        uf = gravity * hh + (uh**2) / 2.0
        return [hf, uf]

    @staticmethod
    def ShallowWaterFrictionSource(hh, uh, visc=1e-6, eps=1e-3, gravity=9.81, inc=0.0, **kwargs):
        # Cálculo do Reynolds
        Re_matrix = 4 * uh * hh / visc
        Re_flat = Re_matrix.flatten()
        hh_flat = hh.flatten()
        uh_flat = uh.flatten()
        Cf_flat = np.zeros_like(Re_flat)
        
        for i, re in enumerate(Re_flat):
            if uh_flat[i] == 0.:
                cf = 0.
            else:
                if re <= 2000.:
                    cf = 16.0 / re  # Equação de Blasius
                else:
                    # Equação de Churchill
                    D = 4 * hh_flat[i]
                    B = (37530.0 / re)**16
                    A = (2.457 * np.log( ((7.0/re)**0.9 + 0.27*(eps/D))**-1 ))**16
                    cf = 2 * ((8.0/re)**12 + (A + B)**-1.5)**(1/12)
            Cf_flat[i] = cf
            
        Cf = Cf_flat.reshape(hh.shape)
        
        # Termo fonte final (atrito + inclinação)
        us = -Cf * (uh**2) / (2 * hh) + gravity * math.sin(inc)
        return [np.zeros_like(us), us] # Conservação de massa não tem fonte, só a de momento

    @staticmethod
    def Euler(rho, rhou, Ener, gamma=1.4, **kwargs):
        """
        Fluxo convectivo da equação de Euler 1D.

        Equação
        --------
        ∂U/∂t + ∂F(U)/∂x = 0

        onde

            U = [ρ, ρu, E]^T

        e

            F(U) =  [ρu, ρu² + p, u(E+p) ]^T

        com

            p = (γ-1)(E - ρu²/(2ρ))

        Parameters
        ----------
        rho : ndarray
            Densidade (ρ).

        rhou : ndarray
            Momento linear (ρu).

        Ener : ndarray
            Energia total (E).

        gamma : float, optional
            Razão dos calores específicos.

        Returns
        -------
        list[ndarray]
            Fluxo conservativo

                [ρu,
                ρu²+p,
                u(E+p)].
        """
        pres = (gamma - 1.0) * (Ener - 0.5 * (rhou**2) / rho)
        
        rhof = rhou.copy()
        rhouf = (rhou**2) / rho + pres
        Enerf = (Ener + pres) * rhou / rho
        
        return [rhof, rhouf, Enerf]

# Exemplo de como o Solver chamaria o fluxo:
# fluxos = PhysicsFluxes.ShallowWater(h_fisico, u_fisico, gravity=9.8)