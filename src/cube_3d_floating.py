import numpy as np
from tqdm import tqdm
import pyvista as pv

import plotting
from transforms import Lq, Rq, H, Aq


dt = 0.001
g = 9.81
m = 10  # mass of the particle in kg
I = np.eye(3)  # inertia matrix
dt = 0.001  # timestep size
mu = 0.1  # coefficient of friction
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


def dynamics_ct(X, U):
    # SE(3) nonlinear dynamics
    # Unpack state vector
    r_w = X[0:3]  # W frame
    Q = X[3:7]  # B to W
    v_w = X[7:10]  # W frame
    ω_b = X[10:13]  # B frame
    F_w = U[0:3]  # W frame
    tau_b = U[3:]  # B frame
    dr = v_w  # rotate v from body to world frame
    dq = 0.5 * Lq(Q) @ H @ ω_b
    dv = 1 / m * F_w
    dω = np.linalg.solve(I, tau_b - np.cross(ω_b, I @ ω_b))
    dX = np.hstack((dr, dq, dv, dω)).T
    return dX


def rk4_normalized(X_k, U_k):
    # RK4 integrator solves for new X
    dynamics = dynamics_ct
    f1 = dynamics(X_k, U_k)
    f2 = dynamics(X_k + 0.5 * dt * f1, U_k)
    f3 = dynamics(X_k + 0.5 * dt * f2, U_k)
    f4 = dynamics(X_k + dt * f3, U_k)
    xn = X_k + (dt / 6.0) * (f1 + 2 * f2 + 2 * f3 + f4)
    xn[3:7] = xn[3:7] / np.linalg.norm(xn[3:7])  # normalize the quaternion term
    return xn


N = 1200  # number of timesteps

n_x = 13  # length of state vector
n_u = 6  # length of control vector
X_0 = np.zeros(n_x)

p_0 = np.array([0, 0, 3.0])
Q_0 = np.array([1, 0, 0, 0])
v_0 = np.array([1, 0, 0])
ω_0 = np.array([2.0, 4.0, 6.0])

X_0[:3] = p_0
X_0[3:7] = Q_0
X_0[7:10] = v_0
X_0[10:13] = ω_0

X_hist = np.zeros((N, n_x))  # array of state vectors for each timestep
Fx_hist = np.zeros(N)  # array of x GRF forces for each timestep
Fz_hist = np.zeros(N)  # array of z GRF forces for each timestep
X_hist[0, :] = X_0
U_hist = np.zeros((N - 1, n_u))  # array of control vectors for each timestep

for k in range(N - 1):
    X_hist[k + 1, :] = rk4_normalized(X_hist[k, :], U_hist[k, :])


name = "cube_3d_floating"
hists = {
    "x (m)": X_hist[:, 0],
    "y (m)": X_hist[:, 1],
    "z (m)": X_hist[:, 2],
}
plotting.plot_hist(hists, name)


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
fps = 30.0
speed = 1  # x real time
plotter.open_gif("results/" + name + ".gif", fps=fps, subrectangles=True)
frames = int(fps * speed)
j = 0
for k in tqdm(range(N)[::frames]):
    r_c = kin_corners(X_hist[k, :])
    text_obj.input = "t = " + "{:.2f}".format(round(k * dt, 2)) + "s"
    mesh.points = r_c
    plotter.write_frame()

# Closes and finalizes movie
plotter.close()
