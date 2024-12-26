import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os_utils


def animate(
    x_hist: np.ndarray,
    z_hist: np.ndarray,
    dt: float,
    name: str,
    xlim: list = [0, 2],
    ylim: list = [0, 2],
    frames=30,
):
    # generate animation
    path_dir_imgs, path_dir_gif = os_utils.prep_animation()
    j = 0
    N = np.shape(x_hist)[0]
    print("Animating...")
    for k in tqdm(range(N - 1)[::frames]):
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.title("Position vs Time")
        plt.xlabel("x (m)")
        plt.ylabel("z (m)")
        plt.scatter(x_hist[k], z_hist[k])
        plt.text(
            0.01,
            0,
            "t = " + "{:.2f}".format(round(k * dt, 2)) + "s",
            ha="left",
            va="bottom",
            transform=plt.gca().transAxes,
        )
        plt.savefig(path_dir_imgs + "/" + str(j).zfill(4) + ".png")
        plt.close()
        j += 1

    os_utils.convert_gif(
        path_dir_imgs=path_dir_imgs, path_dir_output=path_dir_gif, file_name=name
    )


def plot_2d_hist(
    hists: dict,
    N: int,
    name: str,
):
    # plot w.r.t. time
    fig, axs = plt.subplots(len(hists), sharex="all")
    fig.suptitle(name)
    plt.xlabel("timesteps")
    i = 0
    for key, value in hists.items():
        axs[i].plot(range(len(value)), value)
        axs[i].set_ylabel(key)
        i += 1
    fig = plt.gcf()
    fig.tight_layout()
    fig.set_size_inches(10, 10)
    plt.savefig("results/" + name + ".png")
    plt.close()
