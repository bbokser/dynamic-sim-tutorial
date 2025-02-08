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
        LQ = cs.SX(4, 4)
    LQ[0, 0] = Q[0]
    LQ[0, 1:4] = -Q[1:4].T
    LQ[1:4, 0] = Q[1:4]
    LQ[1:4, 1:4] = Q[0] * np.eye(3) + hat(Q[1:4])
    return LQ


def Rq(Q):
    if isinstance(Q, np.ndarray):
        RQ = np.zeros((4, 4))
    else:
        RQ = cs.SX(4, 4)
    RQ[0, 0] = Q[0]
    RQ[0, 1:4] = -Q[1:4].T
    RQ[1:4, 0] = Q[1:4]
    RQ[1:4, 1:4] = Q[0] * np.eye(3) - hat(Q[1:4])
    return RQ


def Aq(Q):
    # rotation matrix from quaternion
    return H.T @ Lq(Q) @ Rq(Q).T @ H


n_a = 13  # length of state vector
n_u = 8  # length of control vector
m = 10  # mass of the cube in kg
I = np.eye(3) * 0.1  # inertia matrix
dt = 0.001  # timestep size
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
    if isinstance(X, np.ndarray):
        r_c = np.zeros((8, 3))
    else:
        r_c = cs.SX(8, 3)
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
    F_w = cs.SX(3, 1)  # force in W frame
    tau_b = cs.SX(3, 1)  # torque in B frame

    # rotation matrix, body to world frame
    A = Aq(Q)
    # iterate through corners to calculate torque acting on body
    F_c_w = cs.SX(8, 3)
    for i in range(n_u):
        F_c_w[i, 2] = U[i]
        F_w += F_c_w[i, :].T
        # add body frame torque due to body frame force
        tau_b += cs.cross(r_c_b[i, :], A.T @ F_c_w[i, :].T)

    F_w += np.array([0, 0, -9.81]) * m  # gravity

    dr = v_w
    dq = 0.5 * Lq(Q) @ H @ ω_b
    dv = 1 / m * F_w
    dω = cs.solve(I, tau_b - cs.cross(ω_b, I @ ω_b))
    dX = cs.vertcat(dr, dq, dv, dω)
    return dX


def rk4_normalized(dynamics: Callable, X_k: cs.SX, U_k: cs.SX) -> cs.SX:
    # RK4 integrator solves for new X
    f1 = dynamics(X_k, U_k)
    f2 = dynamics(X_k + 0.5 * dt * f1, U_k)
    f3 = dynamics(X_k + 0.5 * dt * f2, U_k)
    f4 = dynamics(X_k + dt * f3, U_k)
    X_n = X_k + (dt / 6.0) * (f1 + 2 * f2 + 2 * f3 + f4)
    X_n[3:7] = X_n[3:7] / cs.norm_2(X_n[3:7])  # normalize the quaternion term
    return X_n


def euler_semi_implicit(
    dynamics: Callable, X_k: cs.SX, U_k: cs.SX, X_k1: cs.SX
) -> cs.SX:
    # semi-implicit Euler integrator solves for new X
    X_k_semi = cs.SX.zeros(n_a)
    X_k_semi[:7] = X_k[:7]
    X_k_semi[7:] = X_k1[7:]
    X_n = X_k + dt * dynamics(X_k_semi, U_k)
    X_n[3:7] = X_n[3:7] / cs.norm_2(X_n[3:7])  # normalize the quaternion term
    return X_n


# initialize casadi variables
Xk1 = cs.SX.sym("Xk1", n_a)  # X(k+1), state at next timestep
F = cs.SX.sym("F", n_u)  # force at each corner
s = cs.SX.sym("s", n_u)  # slack variable
X = cs.SX.sym("X", n_a)  # X(k), state

obj = s.T @ s

constr = []  # init constraints
# constr = cs.vertcat(constr, rk4_normalized(dynamics_ct, X, F) - Xk1)
constr = cs.vertcat(constr, euler_semi_implicit(dynamics_ct, X, F, Xk1) - Xk1)

# quaternion normalization
# constr = cs.vertcat(constr, cs.norm_2(Xk1[3:7]) ** 2 - 1)

# stay above the ground
z_c_w = kin_corners(Xk1)[:, 2]  # corner heights
constr = cs.vertcat(constr, z_c_w)

# relaxed complementarity aka compl. slackness
constr = cs.vertcat(constr, s - F * z_c_w)  # ground penetration

opt_variables = cs.vertcat(Xk1, F, s)
lcp = {"x": opt_variables, "p": X, "f": obj, "g": constr}
opts = {
    "print_time": 0,
    "ipopt.print_level": 0,
    "ipopt.tol": 1e-8,
    "ipopt.max_iter": 1500,
}
solver = cs.nlpsol("S", "ipopt", lcp, opts)

n_var = np.shape(opt_variables)[0]
n_g = np.shape(constr)[0]

# variable bounds
ubx = [1e10] * n_var
lbx = [0] * n_var
lbx[:n_a] = [-1e10] * n_a  # state can be negative

# constraint bounds
ubg = [0] * n_g
# ubg[n_a + 1 : n_a + 1 + n_u] = [1e10] * n_u  # set z_c >= 0
# ubg[n_a + 1 + n_u :] = [1e10] * n_u  # set relaxed complementarity >= 0
ubg[n_a : n_a + n_u] = [1e10] * n_u  # set z_c >= 0
ubg[n_a + n_u :] = [1e10] * n_u  # set relaxed complementarity >= 0
lbg = [0] * n_g

# initialize simulation variables
N = 2000  # number of timesteps
X_0 = np.zeros(n_a)
X_0[:3] = np.array([0, 0, 3.0])
X_0[3:7] = np.random.rand(
    4,
)
X_0[3:7] = X_0[3:7] / np.linalg.norm(X_0[3:7])
# X_0[3:7] = np.array([1, 0, 0, 0])
X_0[7:10] = np.array([0, 2, 0])
X_0[10:13] = np.array([0, -1, 1])

X_hist = np.zeros((N, n_a))  # array of state vectors for each timestep
F_hist = np.zeros((N, n_u))  # array of GRF for each timestep
s_hist = np.zeros((N, n_u))  # array of slack var values for each timestep
# energy_hist = np.zeros(N)

X_hist[0, :] = X_0
for k in tqdm(range(N - 1)):
    sol = solver(lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, p=X_hist[k, :])
    X_hist[k + 1, :] = np.reshape(sol["x"][0:n_a], (-1,))
    F_hist[k] = np.reshape(sol["x"][n_a : n_a + n_u], (-1,))
    s_hist[k] = np.reshape(sol["x"][n_a + n_u :], (-1,))


name = "cube_3d_con_tstep"
hists = {
    "x (m)": X_hist[:, 0],
    "y (m)": X_hist[:, 1],
    "z (m)": X_hist[:, 2],
    "ωx (m)": X_hist[:, 10],
    "ωy (m)": X_hist[:, 11],
    "ωz (m)": X_hist[:, 12],
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
fps = 30
speed = 1  # x real time
plotter.open_gif(
    "results/" + name + ".gif", fps=fps, palettesize=64, subrectangles=True
)
frames = int(fps * speed)
j = 0
for k in tqdm(range(N)[::frames]):
    r_c = kin_corners(X_hist[k, :])
    text_obj.input = "t = " + "{:.2f}".format(round(k * dt, 2)) + "s"
    mesh.points = r_c
    plotter.write_frame()

# Closes and finalizes movie
plotter.close()
