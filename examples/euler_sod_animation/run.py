from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import math

# -----------------------------------------------------------------------------
# 1. IMPORTAÇÕES DA TOOLKIT (Apenas Malha e Integrador Temporal)
# -----------------------------------------------------------------------------
from dg1d_toolkit.dg1d_core import DGSpace1D
from dg1d_toolkit.dg1d_integrators import RKSSP54_Step, RKSSP104_Step
from dg1d_toolkit.dg1d_limiters import SlopeLimiterN, HierarchicalLimiter
from dg1d_toolkit.operators import FluxProjection
from dg1d_toolkit.numericalfluxes import rusanov
from dg1d_toolkit.physicalfluxes import PhysicsFluxes


OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "generated"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# 2. CONFIGURAÇÕES ORIGINAIS
# -----------------------------------------------------------------------------
K = 180         
N = 3           
gamma = 1.4
CFL = 1.0
xmin, xmax = 0.0, 1.0
tmin, FinalTime = 0.0, 0.2
dt = 0.000025
Nsteps = 8000

# -----------------------------------------------------------------------------
# 3. Lh baseado no Original com Cirurgias
# -----------------------------------------------------------------------------
def EulerRHS1D(U_list, t, space):
    rhohat, rhouhat, Enerhat = U_list[0], U_list[1], U_list[2]
    C = np.zeros(space.K+1)
    
    rhoh = np.dot(space.psi, rhohat)
    rhouh = np.dot(space.psi, rhouhat)
    Enerh = np.dot(space.psi, Enerhat)
    
    pres = (gamma-1.0)*(Enerh - 0.5*rhouh**(2)/rhoh)
    cvel = np.sqrt(gamma*pres/rhoh)
    lm = np.abs(rhouh/rhoh) + cvel
    
    lmkl = lm[0, :]
    lmkr = lm[-1, :]
    
    C[1:-1] = np.maximum(lmkr[:-1], lmkl[1:])
    C[0] = lmkl[0]
    C[-1] = lmkr[-1]
    
    rhot = np.zeros((space.Nldof, space.K+2, 2))
    rhout = np.zeros((space.Nldof, space.K+2, 2))
    Enert = np.zeros((space.Nldof, space.K+2, 2))
    
    rhoft = np.zeros((space.Nldof, space.K+2))
    rhouft = np.zeros((space.Nldof, space.K+2))
    Enerft = np.zeros((space.Nldof, space.K+2))
        
    rhot[:, 1:-1, 0] = rhohat.copy()
    rhout[:, 1:-1, 0] = rhouhat.copy()
    Enert[:, 1:-1, 0] = Enerhat.copy()
    
    rhohL = np.dot(space.Flkp1[0, :], rhot[:, 1, 0])
    rhohR = np.dot(space.Frk[0, :], rhot[:, -2, 0])
    rhouhL = np.dot(space.Flkp1[0, :], rhout[:, 1, 0])
    rhouhR = np.dot(space.Frk[0, :], rhout[:, -2, 0])
    EnerhL = np.dot(space.Flkp1[0, :], Enert[:, 1, 0])
    EnerhR = np.dot(space.Frk[0, :], Enert[:, -2, 0])
    
    rhoin = 1.0
    rhouin = 0.0
    pin = 1.0
    Enerin = pin/(gamma-1.0)
    rhoout = 0.125
    rhouout = 0.0
    pout = 0.1
    Enerout = pout/(gamma-1.0)        
    
    rhot[0, 0, 0] = -rhohL + 2.0*rhoin    
    rhot[0, -1, 0] = -rhohR + 2.0*rhoout  
    rhout[0, 0, 0] = -rhouhL + 2.0*rhouin    
    rhout[0, -1, 0] = -rhouhR + 2.0*rhouout  
    pint = -pres[0, 0] + 2.0*pin   
    poutt = -pres[-1, -1] + 2.0*pout   
    Enert[0, 0, 0] = -EnerhL + 2.0*Enerin    
    Enert[0, -1, 0] = -EnerhR + 2.0*Enerout  
    
    # =========================================================================
    # CIRURGIA 1: USANDO A FLUX PROJECTION DO TOOLKIT
    # =========================================================================
    # 1. Empacota os estados atuais (fatia do tempo 0) para mandar pro Toolkit
    ut_list = [rhot[:, :, 0], rhout[:, :, 0], Enert[:, :, 0]]
    
    # 2. Chama a projeção genérica passando a função de física
    ft_list = FluxProjection(space, ut_list, PhysicsFluxes.Euler)
    rhoft, rhouft, Enerft = ft_list[0], ft_list[1], ft_list[2]
    
    # 3. Injeta as condições de contorno físicas nas bordas das matrizes projetadas
    rhoft[0, 0] = rhout[0, 0, 0]
    rhoft[0, -1] = rhout[0, -1, 0]
    
    rhouft[0, 0] = rhout[0, 0, 0]**(2)/rhot[0, 0, 0] + pint
    rhouft[0, -1] = rhout[0, -1, 0]**(2)/rhot[0, -1, 0] + poutt
    
    Enerft[0, 0] = (pint/(gamma-1.0) + 0.5*rhout[0, 0, 0]**(2)/rhot[0, 0, 0] + pint)*rhout[0, 0, 0]/rhot[0, 0, 0]
    Enerft[0, -1] = (poutt/(gamma-1.0) + 0.5*rhout[0, -1, 0]**(2)/rhot[0, -1, 0] + poutt)*rhout[0, -1, 0]/rhot[0, -1, 0]

    # =========================================================================
    # CIRURGIA 2: USANDO O FLUXO DE RUSANOV DO TOOLKIT
    # =========================================================================
    # 1. Integrais de Volume (Derivadas físicas)
    vol_rho  = np.dot(space.S.T, rhoft[:, 1:-1])
    vol_rhou = np.dot(space.S.T, rhouft[:, 1:-1])
    vol_Ener = np.dot(space.S.T, Enerft[:, 1:-1])
    
    # 2. Fluxos Numéricos de Borda (Rusanov)
    flux_rho  = rusanov(space, rhot[:, :, 0], rhoft, C)
    flux_rhou = rusanov(space, rhout[:, :, 0], rhouft, C)
    flux_Ener = rusanov(space, Enert[:, :, 0], Enerft, C)
    
    # 3. Montagem Final (Tudo vetorizado e sem transposições duplas!)
    rhot[:, 1:-1, 1]  = space.InvM[:, None] * space.J[:]**(-1) * (vol_rho + flux_rho)
    rhout[:, 1:-1, 1] = space.InvM[:, None] * space.J[:]**(-1) * (vol_rhou + flux_rhou)
    Enert[:, 1:-1, 1] = space.InvM[:, None] * space.J[:]**(-1) * (vol_Ener + flux_Ener)
    # =========================================================================
    
    rhsrho = rhot[:, 1:-1, 1].copy()
    rhsrhou = rhout[:, 1:-1, 1].copy()
    rhsEner = Enert[:, 1:-1, 1].copy()
    
    return [rhsrho, rhsrhou, rhsEner]
    

# -----------------------------------------------------------------------------
# 5. LOOP PRINCIPAL
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    space = DGSpace1D(K=K, N=N, xmin=xmin, xmax=xmax, quad_type='GL')
    
    # Condição Inicial exata do professor
    rhox = np.ones((space.nip, K))
    rhoux = np.zeros((space.nip, K))
    Enerx = (gamma-1.)**(-1)*np.ones((space.nip, K))
    
    idx = int(math.floor(K/2))
    rhox[:, idx:] = 0.125
    Enerx[:, idx:] = 0.1*Enerx[:, idx:]
    
    def IC(ux):
        invdiag = np.zeros(space.Nldof)
        for i in range(0, space.Nldof):
            invdiag[i] = 1./(sum(space.wi*space.psi[:, i]**2))
        b = np.dot(space.wi*space.psi.T, ux)
        fproj = invdiag*b.T
        return fproj.T.copy()

    U_modal = [IC(rhox), IC(rhoux), IC(Enerx)]

    # ============================================================
    # Histórico temporal
    # ============================================================

    nvar = len(U_modal)
    save_every = 80
    Nsaved = Nsteps // save_every + 1

    history = np.zeros((nvar, space.nip * space.K, Nsaved))
    pressure_history = np.zeros((space.nip * space.K, Nsaved))

    # condição inicial
    for i in range(nvar):
        history[i, :, 0] = np.dot(space.psi, U_modal[i]).T.flatten()


    pressure_history[:, 0] = (gamma - 1.0) * (
            history[2, :, 0] - 0.5 * history[1, :, 0]**2 / history[0, :, 0]
            )

    frame = 1

    
    meu_limitador = SlopeLimiterN(space)
    #meu_limitador = HierarchicalLimiter(space)

    t = tmin
    print("Rodando o código com base no do professor...")
    for n in range(1, Nsteps+1):
        RHS_func = lambda U_list, time: EulerRHS1D(U_list, time, space)
        # Escolha qual integrador usar (o original usava RKSSP54 ou 104)
        U_modal = RKSSP54_Step(U_modal, t, dt, RHS_func, meu_limitador)
        t = t + dt

        if n % save_every == 0:
            for i in range(nvar):
                history[i, :, frame] = np.dot(space.psi, U_modal[i]).T.flatten()

            pressure_history[:, frame] = (gamma - 1.0) * (
                history[2, :, frame]
                - 0.5 * history[1, :, frame]**2 / history[0, :, frame])
            frame += 1


    x = space.xc.T.flatten()

    from dg1d_toolkit.dg1d_postprocessing import create_gif
    # nomes = ["Density", "Momentum", "Energy"]
    # pastas = ["rho_animation", "rhou_animation", "energy_animation"]

    # for i in range(nvar):

    #     create_gif(
    #         x=x,
    #         u_history=history[i],
    #         dt=dt * save_every,
    #         folder=pastas[i],
    #         filename="simulation.gif",
    #         title=nomes[i],
    #         xlabel="x",
    #         ylabel=nomes[i],
    #         xlim=(xmin, xmax),
    #         ylim=None,
    #         duration=80,
    #         plot_kwargs={
    #             "lw": 2,
    #         },
    #     )
    create_gif(
        x=x,
        u_history=pressure_history,
        dt=dt * save_every,
        folder=OUTPUT_DIR / "pressure_animation",
        filename="simulation.gif",
        title="Pressure",
        xlabel="x",
        ylabel="Pressure",
        xlim=(xmin, xmax),
        ylim=(0.0, 1.2),
        duration=80,
    )

    # -------------------------------------------------------------------------
    # 6. SOLUÇÃO ANALÍTICA E PLOTS
    # -------------------------------------------------------------------------
    # PrL = 1.0; PrR = 0.1; rhoL = 1.0; rhoR = 0.125
    # uL = 0.0; uR = 0.0; t_end = 0.2; mu = 0.35; 
    
    # gammaA = gamma - 1.0
    # gammaB = 1/gammaA
    # gammaC = gamma + 1.0
    
    # PRL = PrR/PrL
    # cR = np.sqrt(gamma * PrR/rhoR)
    # cL = np.sqrt(gamma * PrL/rhoL)
    # CRL = cR/cL
    # machL = (uL - uR)/cL
    # def func(p34):
    #     wortel = np.sqrt(2. * gamma * (gammaA + gammaC * p34))
    #     yy = (gammaA * CRL * (p34 - 1.)) / wortel
    #     yy = (1. + machL * gammaA/2. - yy )**(2. * gamma/gammaA )
    #     y = yy/p34 - PRL
    #     return y
        
    # p34 = fsolve(func, 3.0)[0]
    # p3 = p34 * PrR
    # alpha = gammaC/gammaA
    # rho3 = rhoR * (1. + alpha * p34)/(alpha + p34)
    # rho2 = rhoL * (p34 * PrR/PrL )**(1/gamma)
    # u2 = uL-uR +(2./gammaA)*cL*(1. - (p34 * PrR/PrL)**(gammaA/(2. * gamma)))
    # c2 = np.sqrt(gamma * p3/ rho2)
    
    # spos = (0.5 + t_end * cR *np.sqrt(gammaA/(2. * gamma) + gammaC/(2. * gamma ) * p34) + t_end * uR)
    # conpos = 0.5 + u2 * t_end + t_end * uR 
    # pos1 = 0.5 + (uL - cL) * t_end 
    # pos2 = 0.5 + (u2 + uR - c2) * t_end 
    
    # xgrid = space.xc.T.flat
    # PrE = np.zeros((1, len(xgrid)))
    # uE= np.zeros((1, len(xgrid)))
    # rhoE = np.zeros((1, len(xgrid)))
    
    # xgrid = np.matrix(xgrid)
    # for i in range(0, xgrid.size ):
    #     if xgrid[0, i] <= pos1:
    #         PrE[0, i], rhoE[0, i], uE[0, i] = PrL, rhoL, uL
    #     elif xgrid[0, i] <= pos2:
    #         PrE[0, i] = (PrL*(1. + (pos1 - xgrid[0, i])/(cL * alpha * t_end ))**(2. * gamma/gammaA ))
    #         rhoE[0, i] = (rhoL*(1+(pos1 - xgrid[0, i])/(cL * alpha * t_end ))**(2./gammaA ))
    #         uE[0, i] = uL + (2./gammaC)*(xgrid[0, i] - pos1)/t_end
    #     elif xgrid[0, i] <= conpos:
    #         PrE[0, i], rhoE[0, i], uE[0, i] = p3, rho2, u2 + uR
    #     elif xgrid[0, i] <= spos:
    #         PrE[0, i], rhoE[0, i], uE[0, i] = p3, rho3, u2 + uR
    #     else:
    #         PrE[0, i], rhoE[0, i], uE[0, i] = PrR, rhoR, uR

    # # Converte Modos para o Espaço Físico
    # rho_final = np.dot(space.psi, U_modal[0]).T.reshape((space.nip*K))
    # rhou_final = np.dot(space.psi, U_modal[1]).T.reshape((space.nip*K))
    # Ener_final = np.dot(space.psi, U_modal[2]).T.reshape((space.nip*K))
    
    # # Plota Densidade
    # plt.figure(1, figsize=(8, 6), dpi=100)
    # plt.grid(True)
    # plt.plot(np.asarray(xgrid)[0], rhoE[0, :], 'r-', label='Analítica')
    # plt.plot(np.asarray(xgrid)[0], rho_final, 'b--', label='DG1D Toolkit')
    # plt.title("Densidade (Progresso)")
    # plt.legend()
    # plt.ylim(0.0, 1.2)
    # plt.xlim(0.0, 1.0)
    
    # # Plota Velocidade
    # plt.figure(2, figsize=(8, 6), dpi=100)
    # plt.grid(True)
    # plt.plot(np.asarray(xgrid)[0], uE[0, :], 'r-', label='Analítica')
    # plt.plot(np.asarray(xgrid)[0], rhou_final/rho_final, 'b--', label='DG1D Toolkit')
    # plt.title("Velocidade (Progresso)")
    # plt.legend()
    # plt.ylim(-0.2, 1.0)
    # plt.xlim(0.0, 1.0)
    
    # # Plota Pressão
    # plt.figure(3, figsize=(8, 6), dpi=100)
    # plt.grid(True)
    # plt.plot(np.asarray(xgrid)[0], PrE[0, :], 'r-', label='Analítica')
    # plt.plot(np.asarray(xgrid)[0], (gamma-1.0)*(Ener_final - 0.5*rhou_final**(2)/rho_final), 'b--', label='DG1D Toolkit')
    # plt.title("Pressão (Progresso)")
    # plt.legend()
    # plt.ylim(0.0, 1.2)
    # plt.xlim(0.0, 1.0)
    
    # plt.show()
