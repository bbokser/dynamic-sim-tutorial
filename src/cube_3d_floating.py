from collections.abc import Callable

import numpy as np
import pyvista as pv
from tqdm import tqdm

import plotting
from transforms import Aq, H, Lq

# mass of the particle in kg
MASS = 10
# inertia matrix
INERTIA = np.eye(3) * 6
# inertia matrix inverse
I_INV = np.linalg.inv(INERTIA)
# timestep size
DT = 0.001
# body frame locations of the 8 corners of the cube
C_B = np.array(
    (
        [-1, -1, -1],
        [1, -1, -1],
        [-1, 1, -1],
        [1, 1, -1],
        [-1, -1, 1],
        [1, -1, 1],
        [-1, 1, 1],
        [1, 1, 1],
    )
)


def get_energy(X: np.ndarray) -> float:
    """
    Calculate total energy in system
    :param X: state vector
    """
    v_w = X[7:10]  # W frame
    ω_b = X[10:13]  # B frame
    return 0.5 * MASS * np.linalg.norm(v_w) ** 2 + 0.5 * ω_b.T @ INERTIA @ ω_b


def plot_energy(X_hist: np.ndarray, name: str) -> None:
    N = np.shape(X_hist)[0]
    energy_hist = np.zeros(N)
    for k in tqdm(range(N - 1), desc="Calculating energy"):
        energy_hist[k] = get_energy(X_hist[k, :])

    hists_2 = {
        "energy (J)": energy_hist,
    }
    plotting.plot_hist(hists_2, name + " energy")


def kin_corners(X: np.ndarray) -> np.ndarray:
    """
    Get world frame locations of the 8 corners of the cube

    :param X: state vector
    """
    ones_nc = np.ones((8, 1))
    # position of cube in world frame
    r_w = X[0:3].reshape((-1, 1))
    # body to world frame quaternion
    Q = X[3:7]
    # rotation matrix
    A = Aq(Q)
    # rotate C_B and add r_w
    C_W = (A @ C_B.T).T + ones_nc @ r_w.T
    return C_W


def dynamics_floating_ct(X: np.ndarray, U: np.ndarray) -> np.ndarray:
    """
    Continuous-time SE(3) nonlinear dynamics

    :param X: state vector
    :param U: control vector
    """
    # r_w = X[0:3]  # W frame
    Q = X[3:7]  # B to W
    v_w = X[7:10]  # W frame
    ω_b = X[10:13]  # B frame
    F_w = U[0:3]  # W frame
    tau_b = U[3:]  # B frame
    dr = v_w
    dq = 0.5 * Lq(Q) @ H @ ω_b
    dv = 1 / MASS * F_w
    # dω = np.linalg.solve(INERTIA, tau_b - np.cross(ω_b, INERTIA @ ω_b))
    dω = I_INV @ (tau_b - np.cross(ω_b, INERTIA @ ω_b))
    dX = np.hstack((dr, dq, dv, dω)).T
    return dX


def rk4_normalized(dynamics: Callable, X_k: np.ndarray, U_k: np.ndarray) -> np.ndarray:
    """
    RK4 integrator

    :param dynamics: dynamics function
    :param X_k: state vector at step k
    :param U_k: control vector at step k
    """
    f1 = dynamics(X_k, U_k)
    f2 = dynamics(X_k + 0.5 * DT * f1, U_k)
    f3 = dynamics(X_k + 0.5 * DT * f2, U_k)
    f4 = dynamics(X_k + DT * f3, U_k)
    xn = X_k + (DT / 6.0) * (f1 + 2 * f2 + 2 * f3 + f4)
    xn[3:7] = xn[3:7] / np.linalg.norm(xn[3:7])  # normalize the quaternion term
    return xn


def animate_cube(X_hist: np.ndarray, name: str) -> None:
    """
    Convert state hist into gif

    :param X_hist: state history
    """
    N = np.shape(X_hist)[0]
    mesh = pv.Box()
    mesh_plane = pv.Plane(i_size=20, j_size=20, i_resolution=1, j_resolution=1)
    text_obj = pv.Text("t = 0.00 s", position=[0, 0])
    text_obj.prop.color = "black"
    text_obj.prop.font_size = 20
    plotter = pv.Plotter(notebook=False, off_screen=True)
    plotter.add_mesh(mesh, show_edges=True, color="white")
    plotter.add_mesh(mesh_plane, show_edges=True, color="white")
    plotter.camera.zoom(1.5)
    plotter.add_actor(text_obj)
    fps = 30.0
    speed = 1  # x real time
    plotter.open_gif(
        "results/" + name + ".gif", fps=fps, palettesize=64, subrectangles=True
    )
    frames = int(fps * speed)
    for k in tqdm(range(N)[::frames], desc="Generating gif"):
        r_c = kin_corners(X_hist[k, :])
        text_obj.input = "t = " + f"{round(k * DT, 2):.2f}" + "s"
        mesh.points = r_c
        plotter.write_frame()

    # Closes and finalizes movie
    plotter.close()


def main():
    N = 1200  # number of timesteps
    n_x = 13  # length of state vector
    n_u = 6  # length of control vector

    # initialize starting state
    X_0 = np.zeros(n_x)
    # position
    X_0[:3] = np.array([0, 0, 3.0])
    # quaternion
    X_0[3:7] = np.array([1, 0, 0, 0])
    # linear velocity
    X_0[7:10] = np.array([1, 0, 0])
    # angular velocity
    X_0[10:13] = np.array([2.0, 4.0, 6.0])

    # array of state vectors for each timestep
    X_hist = np.zeros((N, n_x))
    X_hist[0, :] = X_0
    # array of control vectors for each timestep
    U_hist = np.zeros((N - 1, n_u))

    for k in range(N - 1):
        X_hist[k + 1, :] = rk4_normalized(
            dynamics_floating_ct, X_hist[k, :], U_hist[k, :]
        )

    name = "cube_3d_floating"
    hists = {
        "x (m)": X_hist[:, 0],
        "y (m)": X_hist[:, 1],
        "z (m)": X_hist[:, 2],
    }
    plotting.plot_hist(hists, name)
    animate_cube(X_hist, name)
    plot_energy(X_hist, name)


if __name__ == "__main__":
    main()
