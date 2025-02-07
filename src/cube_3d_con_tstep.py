import numpy as np
from tqdm import tqdm
import pyvista as pv
import casadi as cs
from collections.abc import Callable

import plotting

H = np.zeros((4, 3))
H[1:4, 0:4] = np.eye(3)


def hat(w):
    # skew-symmetric
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])


def Lq(Q):
    if isinstance(Q, np.ndarray):
        LQ = np.zeros((4, 4))
    else:
        LQ = cs.SX.zeros((4, 4))
    LQ[0, 0] = Q[0]
    LQ[0, 1:4] = -Q[1:4].T
    LQ[1:4, 0] = Q[1:4]
    LQ[1:4, 1:4] = Q[0] * np.eye(3) + hat(Q[1:4])
    return LQ


def Rq(Q):
    if isinstance(Q, np.ndarray):
        RQ = np.zeros((4, 4))
    else:
        RQ = cs.SX.zeros((4, 4))
    RQ[0, 0] = Q[0]
    RQ[0, 1:4] = -Q[1:4].T
    RQ[1:4, 0] = Q[1:4]
    RQ[1:4, 1:4] = Q[0] * np.eye(3) - hat(Q[1:4])
    return RQ


def Aq(Q):
    # rotation matrix from quaternion
    return H.T @ Lq(Q) @ Rq(Q).T @ H


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
n_a = 13  # length of state vector
n_u = 8  # length of control vector


def kin_corners(X: np.ndarray) -> np.ndarray:
    # world frame locations of the 8 corners of the cube
    if isinstance(X, np.ndarray):
        r_c = np.zeros((8, 3))
    else:
        r_c = cs.SX.zeros((8, 3))
    r_w = X[0:3]  # W frame
    Q = X[3:7]  # B to W
    A = Aq(Q)  # rotation matrix
    for i in range(8):
        r_c[i, :] = A @ r_c_b[i, :] + r_w
    return r_c


def dynamics_ct(X: cs.SX, U: cs.SX) -> cs.SX:
    # SE(3) nonlinear dynamics
    # Unpack state vector
    r_w = X[0:3]  # W frame
    Q = X[3:7]  # B to W
    v_w = X[7:10]  # W frame
    ω_b = X[10:13]  # B frame
    F_w = cs.SX.zeros(3)
    tau_b = cs.SX.zeros(3)  # torque in B frame

    # rotation matrix, body to world frame
    A = Aq(Q)
    # iterate through corners to calculate torque acting on body
    for i in range(8):
        F_i_w = cs.SX.zeros(3)
        F_i_w[2] = U[i]
        F_w += F_i_w
        # transform force from world frame back to body frame
        F_i_b = A.T @ F_i_w
        # add body frame torque due to body frame force
        tau_b += cs.cross(r_c_b[i, :], F_i_b)

    F_w += F_g_w  # add gravity

    dr = v_w
    dq = 0.5 * Lq(Q) @ H @ ω_b
    dv = 1 / m * F_w
    dω = cs.solve(I, tau_b - cs.cross(ω_b, I @ ω_b))
    dX = cs.vertcat(dr, dq, dv, dω)
    return dX


def rk4_normalized(dynamics: Callable, X_k: np.ndarray, U_k: np.ndarray) -> np.ndarray:
    # RK4 integrator solves for new X
    f1 = dynamics(X_k, U_k)
    f2 = dynamics(X_k + 0.5 * dt * f1, U_k)
    f3 = dynamics(X_k + 0.5 * dt * f2, U_k)
    f4 = dynamics(X_k + dt * f3, U_k)
    X_n = X_k + (dt / 6.0) * (f1 + 2 * f2 + 2 * f3 + f4)
    X_n[3:7] = X_n[3:7] / cs.norm_2(X_n[3:7])  # normalize the quaternion term
    return X_n


# initialize casadi variables
Xk1 = cs.SX.sym("Xk1", n_a)  # X(k+1), state at next timestep
F = cs.SX.sym("F", n_u)  # force
s = cs.SX.sym("s", n_u)  # slack variable
X = cs.SX.sym("X", n_a)  # state

obj = s.T @ s

constr = []  # init constraints
constr = cs.vertcat(constr, cs.SX(rk4_normalized(dynamics_ct, X, F) - Xk1))

# quaternion normalization
constr = cs.vertcat(constr, cs.norm_2(Xk1[3:7]) ** 2 - 1)

# stay above the ground
z_c_w = kin_corners(Xk1)[:, 2]
constr = cs.vertcat(constr, z_c_w)

# relaxed complementarity aka compl. slackness
constr = cs.vertcat(constr, cs.SX(s - F * z_c_w))  # ground penetration

opt_variables = cs.vertcat(Xk1, F, s)
lcp = {"x": opt_variables, "p": X, "f": obj, "g": constr}
opts = {
    "print_time": 0,
    "ipopt.print_level": 0,
    "ipopt.tol": 1e-6,
    "ipopt.max_iter": 500,
}
solver = cs.nlpsol("S", "ipopt", lcp, opts)

n_var = np.shape(opt_variables)[0]
n_par = n_a  # np.shape(parameters)[0]
n_g = np.shape(constr)[0]

# variable bounds
ubx = [1e10] * n_var
lbx = [-1e10] * n_var
lbx[n_a : n_a + n_u] = [0] * n_u  # set F positive only
lbx[n_a + n_u :] = [0] * n_u  # set slack variables >= 0

# constraint bounds
ubg = [0] * n_g
ubg[n_a + 1 : n_a + 1 + n_u] = [1e10] * n_u  # set z_c >= 0
ubg[n_a + 1 + n_u :] = [1e10] * n_u  # set relaxed complementarity >= 0
lbg = [0] * n_g

# initialize simulation variables
N = 1000  # number of timesteps
X_0 = np.zeros(n_a)
p_0 = np.array([0, 0, 3.0])
Q_0 = np.array([1, 0, 0, 0])
v_0 = np.array([2, 0, 0])
ω_0 = np.array([0, 0, 0])
X_0[:3] = p_0
X_0[3:7] = Q_0
X_0[7:10] = v_0
X_0[10:13] = ω_0

X_hist = np.zeros((N, n_a))  # array of state vectors for each timestep
F_hist = np.zeros((N, n_u))  # array of GRF for each timestep
s_hist = np.zeros((N, n_u))  # array of slack var values for each timestep
energy_hist = np.zeros(N)

X_hist[0, :] = X_0
p_values = np.zeros(n_par)
x0_values = np.zeros(n_var)
for k in tqdm(range(N - 1)):
    # print("timestep = ", k)
    p_values = X_hist[k, :]
    x0_values[:n_a] = X_hist[k, :]
    sol = solver(lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, p=p_values, x0=x0_values)
    X_hist[k + 1, :] = np.reshape(sol["x"][0:n_a], (-1,))
    F_hist[k] = np.reshape(sol["x"][n_a : n_a + n_u], (-1,))
    s_hist[k] = np.reshape(sol["x"][n_a + n_u :], (-1,))


name = "cube_3d_con_tstep"
hists = {
    "x (m)": X_hist[:, 0],
    "y (m)": X_hist[:, 1],
    "z (m)": X_hist[:, 2],
    "F_c (N)": F_hist,
    "s": s_hist,
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
speed = 0.25  # x real time
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
