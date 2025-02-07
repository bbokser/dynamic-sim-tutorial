import numpy as np
from tqdm import tqdm
import pyvista as pv

# from mpl_toolkits.mplot3d import Axes3D

import plotting
from transforms import Lq, Rq, H, Aq, quat_to_axis_angle


g = 9.81
m = 10  # mass of the particle in kg
I = np.eye(3) * 0.1  # inertia matrix
dt = 0.001  # timestep size
mu = 0.1  # coefficient of friction
# Gravitational force in world frame
F_g_w = np.array([0, 0, -g]) * m

l = 1  # half length of cube
# body frame locations of the 8 corners of the cube
r_c_b = np.array(
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


def kin_corners(X: np.ndarray) -> np.ndarray:
    # world frame locations of the 8 corners of the cube
    r_c = np.zeros((8, 3))
    r_w = X[0:3]  # W frame
    Q = X[3:7]  # B to W
    A = Aq(Q)  # rotation matrix
    for i in range(8):
        r_c[i, :] = A @ r_c_b[i, :] + r_w
    return r_c


def get_forces(z: float, dr: np.ndarray) -> np.ndarray:
    v_tang = dr[:2]  # tangential vel. (x and y)
    dz = dr[2]  # z vel
    k = 0.01  # spring constant
    b = 0.1  # damping constant
    amp = 6000  # desired max force
    c = amp * 0.5 / k
    distance_fn = -c * np.tanh(z * 100) + c
    F_spring = k * distance_fn
    F_damper = -b * dz * distance_fn * (np.sign(dz) > 0)
    Fz = F_spring + F_damper
    Fz = np.clip(Fz, 0, amp)
    if np.linalg.norm(v_tang) == 0:
        F_tang = np.zeros(2)
    else:
        F_tang = -mu * Fz * v_tang / np.linalg.norm(v_tang)
        a_tang = F_tang / m
        # don't let friction change the object's direction--that's not possible
        a_tang = np.clip(a_tang * dt, -v_tang, 0) / dt
        F_tang = a_tang * m  # update Fx
    return np.hstack((F_tang, Fz))


def dynamics_ct(X: np.ndarray, U: np.ndarray) -> np.ndarray:
    # SE(3) nonlinear dynamics
    # Unpack state vector
    r_w = X[0:3]  # W frame
    Q = X[3:7]  # B to W
    v_w = X[7:10]  # W frame
    ω_b = X[10:13]  # B frame
    F_w = U[0:3]  # W frame
    tau_b = U[3:]  # B frame
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
ω_0 = np.array([0, 0, 0])

X_0[:3] = p_0
X_0[3:7] = Q_0
X_0[7:10] = v_0
X_0[10:13] = ω_0

X_hist = np.zeros((N, n_x))  # array of state vectors for each timestep
F_hist = np.zeros((N, 3))  # array of GRF vectors for each timestep
tau_hist = np.zeros((N, 3))  # array of reaction torque vectors for each timestep
energy_hist = np.zeros(N)

X_hist[0, :] = X_0
U_hist = np.zeros((N - 1, n_u))  # array of control vectors for each timestep

for k in range(N - 1):
    r_w = X_hist[k, 0:3]
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
        dr_i_w = A @ np.cross(ω_b, r_c_b[i, :]) + r_w
        # get world frame force on corner
        F_c_w = get_forces(z_i, dr_i_w)
        # add to world frame force
        F_hist[k, :] += F_c_w
        # transform force from world frame back to body frame
        F_c_b = A.T @ F_c_w
        # add body frame torque due to body frame force
        tau_hist[k, :] += np.cross(r_c_b[i, :], F_c_b)
    # add gravity
    F_hist[k, :] += F_g_w
    X_hist[k + 1, :] = rk4_normalized(
        X_hist[k, :], np.hstack((F_hist[k, :], tau_hist[k, :]))
    )
    energy_hist[k] = (
        0.5 * m * np.linalg.norm(v_w) ** 2 + m * g * r_w[2] + 0.5 * ω_b.T @ I @ ω_b
    )
    # if np.isnan(X_hist[k, 2]):
    #     break


name = "cube_3d_con_smooth"
hists = {
    "z (m)": X_hist[:, 2],
    "dx (m/s)": X_hist[:, 7],
    "dy (m/s)": X_hist[:, 8],
    "Fx (N)": F_hist[:, 0],
    "Fy (N)": F_hist[:, 1],
    "Fz (N)": F_hist[:, 2],
    # "tau_x (Nm)": tau_hist[:, 0],
    # "tau_y (Nm)": tau_hist[:, 1],
    # "tau_z (Nm)": tau_hist[:, 2],
}
plotting.plot_hist(hists, name)


hists_2 = {
    "energy (J)": energy_hist,
}
plotting.plot_hist(hists_2, name + " energy")

# ---#
mesh = pv.Box()
mesh_plane = pv.Plane(i_size=20, j_size=20, i_resolution=1, j_resolution=1)
text_obj = pv.Text(
    "t = 0.00 s",
    position=[0, 0],
)
text_obj.prop.color = "black"
text_obj.prop.font_size = 20
plotter = pv.Plotter(notebook=False, off_screen=True)
plotter.add_mesh(mesh, show_edges=True, color="white")
plotter.add_mesh(mesh_plane, show_edges=True, color="white")
plotter.add_actor(text_obj)
# plotter.camera.zoom(0.5)
plotter.open_gif("results/" + name + ".gif", fps=30, subrectangles=True)
j = 0
frames = 20
for k in tqdm(range(N)[::frames]):
    r_c = kin_corners(X_hist[k, :])
    text_obj.input = "t = " + "{:.2f}".format(round(k * dt, 2)) + "s"
    mesh.points = r_c
    # Q_k = X_hist[k, 3:7]
    # vector, angle = quat_to_axis_angle(Q_k)
    # mesh.rotate_vector(vector=vector, angle=angle)  # , point=axes.origin)
    # Write a frame. This triggers a render.
    plotter.write_frame()

# Closes and finalizes movie
plotter.close()
