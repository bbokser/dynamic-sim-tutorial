import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import plotting
from transforms import Lq, Rq, H, Aq
import os_utils


dt = 0.001
g = 9.81
m = 10  # mass of the particle in kg
I = np.eye(3) * 0.01  # inertia matrix
dt = 0.001  # timestep size
mu = 0.1  # coefficient of friction
# Gravitational force in world frame
F_g_w = np.array([0, 0, -g]) * m

l = 1  # half length of cube
# body frame locations of the 8 corners of the cube
r_c_b = np.array(
    (
        [l, l, l],
        [l, -l, l],
        [-l, -l, l],
        [-l, l, l],
        [l, l, -l],
        [l, -l, -l],
        [-l, -l, -l],
        [-l, l, -l],
    )
)


def kin_corners(X: np.ndarray) -> np.ndarray:
    # world frame locations of the 8 corners of the cube
    r_c = np.zeros((8, 3))
    r_w = X[0:3]  # W frame
    Q = X[3:7]  # B to W
    A = Aq(Q)  # rotation matrix
    for i in range(8):
        r_c[i, :] = A @ r_c_b[i, :] + r_w
    return r_c


def get_grf(z: float, dz: float) -> float:
    k = 0.01
    b = 0.1  # damping constant
    amp = 2e3  # desired max force
    c = amp * 0.5 / k
    distance_fn = -c * np.tanh(z * 5) + c
    F_spring = k * distance_fn
    F_damper = -b * dz * distance_fn
    grf = F_spring + F_damper
    grf = np.clip(grf, 0, amp)
    return grf


def dynamics_ct(X: np.ndarray, U: np.ndarray) -> np.ndarray:
    # SE(3) nonlinear dynamics
    # Unpack state vector
    r_w = X[0:3]  # W frame
    Q = X[3:7]  # B to W
    v_w = X[7:10]  # W frame
    ω_b = X[10:13]  # B frame
    F_w = U[0:3]  # W frame
    tau_b = U[3:]  # B frame

    # add gravity
    F_w += F_g_w

    dr = v_w
    dq = 0.5 * Lq(Q) @ H @ ω_b
    dv = 1 / m * F_w
    dω = np.linalg.solve(I, tau_b - np.cross(ω_b, I @ ω_b))
    dX = np.hstack((dr, dq, dv, dω)).T
    return dX


def rk4_normalized(X_k: np.ndarray, U_k: np.ndarray) -> np.ndarray:
    # RK4 integrator solves for new X
    dynamics = dynamics_ct
    f1 = dynamics(X_k, U_k)
    f2 = dynamics(X_k + 0.5 * dt * f1, U_k)
    f3 = dynamics(X_k + 0.5 * dt * f2, U_k)
    f4 = dynamics(X_k + dt * f3, U_k)
    X_n = X_k + (dt / 6.0) * (f1 + 2 * f2 + 2 * f3 + f4)
    X_n[3:7] = X_n[3:7] / np.linalg.norm(X_n[3:7])  # normalize the quaternion term
    return X_n


N = 3000  # number of timesteps
n_x = 13  # length of state vector
n_u = 6  # length of control vector
X_0 = np.zeros(n_x)

p_0 = np.array([0, 0, 3.0])
Q_0 = np.array([1, 0, 0, 0])
v_0 = np.array([2, 0, 0])
ω_0 = np.array([1, 0.5, 0])

X_0[:3] = p_0
X_0[3:7] = Q_0
X_0[7:10] = v_0
X_0[10:13] = ω_0

X_hist = np.zeros((N, n_x))  # array of state vectors for each timestep
F_hist = np.zeros((N, 3))  # array of GRF vectors for each timestep
tau_hist = np.zeros((N, 3))  # array of reaction torque vectors for each timestep

X_hist[0, :] = X_0
U_hist = np.zeros((N - 1, n_u))  # array of control vectors for each timestep

for k in range(N - 1):
    Q = X_hist[k, 3:7]  # B to W
    v_w = X_hist[k, 7:10]  # W frame
    ω_b = X_hist[k, 10:13]  # B frame
    # rotation matrix, body to world frame
    A = Aq(Q)
    # world frame locations of the 8 corners of the cube
    r_c_w = kin_corners(X_hist[k, :])
    # iterate through corners
    for i in range(8):
        # get height of corner in world frame
        z_i = r_c_w[i, 2]
        # get z aspect of linear velocity of corner in world frame
        dr_w_i = A @ np.cross(ω_b, r_c_b[i, :])
        dz_i = dr_w_i[2]
        # get world frame force on corner
        F_c_w = np.array((0, 0, get_grf(z_i, dz_i)))
        # add to world frame force
        F_hist[k, :] += F_c_w
        # transform force from world frame back to body frame
        F_c_b = A.T @ F_c_w
        # add torque due to body frame force
        tau_hist[k, :] += np.cross(r_c_b[i, :], F_c_b)
    U_hist[k, :3] = F_hist[k, :]
    U_hist[k, 3:] = tau_hist[k, :]
    X_hist[k + 1, :] = rk4_normalized(X_hist[k, :], U_hist[k, :])
    # if np.isnan(X_hist[k, 2]):
    #     break


name = "cube_3d_con_smooth"
hists = {
    "z (m)": X_hist[:, 2],
    "Fz (N)": F_hist[:, 2],
    "tau_x (Nm)": tau_hist[:, 0],
    "tau_y (Nm)": tau_hist[:, 1],
    "tau_z (Nm)": tau_hist[:, 2],
}
plotting.plot_hist(hists, name)


path_dir_imgs, path_dir_gif = os_utils.prep_animation()
j = 0
frames = 20
for k in tqdm(range(N)[::frames]):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    r_c = kin_corners(X_hist[k, :])
    for i in range(8):
        x1 = r_c[i, 0]
        y1 = r_c[i, 1]
        z1 = r_c[i, 2]
        ax.scatter(x1, y1, z1)
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.set_zlim(0, 12)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    plt.title(name)
    ax.text(
        x=0.01,
        y=0,
        z=0,
        s="t = " + "{:.2f}".format(round(k * dt, 2)) + "s",
        ha="left",
        va="bottom",
        transform=plt.gca().transAxes,
    )
    fig.savefig(path_dir_imgs + "/" + str(j).zfill(4) + ".png")
    plt.close()
    j += 1

os_utils.convert_gif(
    path_dir_imgs=path_dir_imgs, path_dir_output=path_dir_gif, file_name=name
)
