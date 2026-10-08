import os
from PIL import Image
import matplotlib.pyplot as plt


def create_gif(x,u_history,dt,folder="animation",filename="simulation.gif",
    xlabel="x",ylabel="u",title="Simulation",ylim=None,xlim=None,
    dpi=120,duration=80,grid=True,
    figsize=(8, 6), plot_kwargs=None,
):
    """
    Gera um GIF a partir do histórico temporal da solução.

    Parameters
    ----------
    x : ndarray
        Coordenadas espaciais.

    u_history : ndarray
        Matriz (Npoints, Nsteps+1) contendo a solução em todos os instantes.

    dt : float
        Passo de tempo.

    folder : str
        Diretório onde serão salvos os frames e o GIF.

    filename : str
        Nome do arquivo GIF.

    xlabel, ylabel : str
        Rótulos dos eixos.

    title : str
        Título base da figura.

    ylim, xlim : tuple or None
        Limites dos eixos.

    dpi : int
        Resolução das imagens.

    duration : int
        Tempo entre quadros em milissegundos.

    grid : bool
        Exibe grade.

    figsize : tuple
        Tamanho da figura.

    plot_kwargs : dict
        Argumentos adicionais para plt.plot().
    """

    os.makedirs(folder, exist_ok=True)
    if plot_kwargs is None:
        plot_kwargs = {"lw": 2}

    fig, ax = plt.subplots(figsize=figsize)
    image_files = []
    nsteps = u_history.shape[1]

    for n in range(nsteps):
        ax.clear()
        ax.plot(x, u_history[:, n], **plot_kwargs)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title}\n$t = {n*dt:.4f}$")

        if xlim is not None:
            ax.set_xlim(xlim)

        if ylim is not None:
            ax.set_ylim(ylim)

        ax.grid(grid)
        fig.tight_layout()
        frame = os.path.join(folder, f"frame_{n:04d}.png")
        fig.savefig(frame, dpi=dpi)
        image_files.append(frame)

    plt.close(fig)
    images = [Image.open(frame) for frame in image_files]
    gif_path = os.path.join(folder, filename)

    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0,
    )

    print(f"GIF salvo em: {gif_path}")