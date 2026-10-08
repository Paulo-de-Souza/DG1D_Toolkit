import numpy as np

def ModifStiffMatrix(space, epst, epsb=None, type_form='Warburton'):
    """
    Calcula a Matriz de Rigidez Modificada para o termo difusivo.
    Puxa a matemática do objeto 'space' de forma automática.
    """
    if type_form == 'Warburton':
        if epsb is None:
            raise ValueError("Para a abordagem 'Warburton', 'epsb' não pode ser vazio.")

        Ssq1 = np.dot(np.sqrt(epst) * space.wi * space.psi.T, space.Dpsi)
        Boundterm = (np.sqrt(epsb[-1]) * space.Frk) - (np.sqrt(epsb[0]) * space.Flk)
        Ssq2 = Boundterm - Ssq1.T

        return Ssq1.T, Ssq2.T

    elif type_form == 'Persson':
        Ssq1 = np.dot(epst * space.wi * space.psi.T, space.Dpsi)
        return Ssq1.T, None
    else:
        raise ValueError("Modelo de matriz modificada não reconhecido!")


def FluxProjection(space, ut_list, physical_flux_func, **kwargs):
    """
    Projeta o fluxo físico no espaço modal através de uma projeção L2
    usando superintegração para evitar erros de aliasing.
    
    Parâmetros:
    - space: objeto DGSpace1D
    - ut_list: lista contendo os estados modais
    - physical_flux_func: função que avalia o fluxo físico (ex: f(u) = u**2 / 2)
    - kwargs: argumentos para a fisica do problema caso o usuario use o PhysicalFluxes da toolkit
    """
    uhat_list = [ut[:, 1:-1] for ut in ut_list]
    
    # Reconstrução da solução física nos pontos de quadratura não-linear
    uh_list = [np.dot(space.nlpsi, uhat) for uhat in uhat_list]   
    
    # Avaliação dos fluxos físicos chamando a função passada pelo usuário
    fh_list = physical_flux_func(*uh_list, **kwargs)   

    if not isinstance(fh_list, (list, tuple)):
        fh_list = [fh_list]

    ProjectionMatrix = space.nlwi * space.nlpsi.T
    ft_list = []

    # Projeção L2 devolvendo arrays com ghost cells (K+2) preenchidas com zero nas bordas
    for fh in fh_list:
        b = np.dot(ProjectionMatrix, fh)
        fhat = space.InvM[:, None] * b
        
        # Aloca array com células fantasmas
        ft = np.zeros((space.Nldof, space.K + 2))
        ft[:, 1:-1] = fhat
        ft_list.append(ft)

    return ft_list