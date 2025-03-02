import numpy as np
from tqdm import tqdm
import pyvista as pv
import casadi as cs
from collections.abc import Callable

import plotting

ϵ = 1e-6


def smoothnorm(x: cs.SX):
    return cs.sqrt(x.T @ x + ϵ * ϵ) - ϵ


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
n_c = 8  # number of contact points on cube
m = 10  # mass of the cube in kg
I = np.eye(3) * 0.1  # inertia matrix
dt = 0.001  # timestep size
mu = 0.1  # coefficient of friction
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


def dynamics_ct(X: cs.SX, F: cs.SX) -> cs.SX:
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
    for i in range(n_c):
        F_w += F[i, :].T
        # add body frame torque due to body frame force
        tau_b += cs.cross(r_c_b[i, :], A.T @ F[i, :].T)

    F_w += np.array([0, 0, -9.81]) * m  # gravity

    dr = v_w
    dq = 0.5 * Lq(Q) @ H @ ω_b
    dv = 1 / m * F_w
    dω = cs.solve(I, tau_b - cs.cross(ω_b, I @ ω_b))
    dX = cs.vertcat(dr, dq, dv, dω)
    return dX


def euler_semi_implicit(
    dynamics: Callable, X_k: cs.SX, U_k: cs.SX, X_k1: cs.SX
) -> cs.SX:
    # semi-implicit Euler integrator solves for new X
    X_k_semi = cs.SX.zeros(n_a)
    X_k_semi[:7] = X_k[:7]
    X_k_semi[7:] = X_k1[7:]
    X_n = X_k + dt * dynamics(X_k_semi, U_k)
    # X_n[3:7] = X_n[3:7] / cs.norm_2(X_n[3:7])  # normalize the quaternion term
    return X_n


# initialize casadi variables
Xk1 = cs.SX.sym("Xk1", n_a)  # X(k+1), state at next timestep
F = cs.SX.sym("F", n_c, 3)  # force vector at each corner
s1 = cs.SX.sym("s1", n_c)  # slack variable 1
s2 = cs.SX.sym("s2", n_c)  # slack variable 2
# lagrange mult for magnitude of ground vel per contact point
lam = cs.SX.sym("lam", n_c)
X = cs.SX.sym("X", n_a)  # X(k), state

obj = s1.T @ s1 + s2.T @ s2

r_c_w_prev = kin_corners(X)  # corner positions at k, 8x3
r_c_w = kin_corners(Xk1)  # corner positions at k+1, 8x3
v_tan = ((r_c_w - r_c_w_prev) / dt)[:, 0:2]  # corner xy velocities, 8x2
z_c_w = r_c_w[:, 2]  # corner heights at k+1, 8x1
F_tan = F[:, :2]  # tangential ground force friction vectors, 8x2
F_z = F[:, 2]  # vertical grfs, 8x1

constr = []  # init constraints

# --- Equality Constraints --- #
# constr = cs.vertcat(constr, rk4_normalized(dynamics_ct, X, F) - Xk1)
constr = cs.vertcat(constr, euler_semi_implicit(dynamics_ct, X, F, Xk1) - Xk1)

# quaternion normalization
constr = cs.vertcat(constr, cs.norm_2(Xk1[3:7]) ** 2 - 1)

for i in range(n_c):
    # max dissipation for each corner
    constr = cs.vertcat(
        constr,
        v_tan[i, :].T + lam[i] * F_tan[i, :].T / (smoothnorm(F_tan[i, :].T) + ϵ),
    )

# --- Inequality Constraints --- #

# interpenetration
constr = cs.vertcat(constr, z_c_w)

for i in range(n_c):
    # primal feasibility friction cone
    constr = cs.vertcat(constr, mu * F_z[i] - smoothnorm(F_tan[i, :].T))

# interpenetration complementarity
constr = cs.vertcat(constr, s1 - F_z * z_c_w)

for i in range(n_c):
    # friction complementarity
    constr = cs.vertcat(
        constr, s2[i] - lam[i] * (mu * F_z[i] - smoothnorm(F_tan[i, :].T))
    )

opt_variables = cs.vertcat(Xk1, F[:, 0], F[:, 1], F[:, 2], s1, s2, lam)
lcp = {"x": opt_variables, "p": X, "f": obj, "g": constr}
opts = {
    "print_time": 0,
    "ipopt.print_level": 0,
    "ipopt.tol": ϵ,
    "ipopt.max_iter": 3000,
}
solver = cs.nlpsol("S", "ipopt", lcp, opts)

n_var = np.shape(opt_variables)[0]
n_g = np.shape(constr)[0]

# variable bounds
ubx = [1e10] * n_var
lbx = [0] * n_var
lbx[:n_a] = [-1e10] * n_a  # state can be negative
lbx[2] = 1  # z pos can't get closer to the ground than 1 m
lbx[n_a : n_a + n_c * 2] = [-1e10] * (n_c * 2)  # Fx and Fy can be negative

# constraint bounds
ubg = [0] * n_g
ubg[n_a + 1 + n_c * 2 :] = [1e10] * (n_c * 4)  # inequality constraints
lbg = [0] * n_g

# initialize simulation variables
N = 2000  # number of timesteps
X_0 = np.zeros(n_a)
X_0[:3] = np.array([0, 0, 3.0])
X_0[3:7] = np.random.rand(
    4,
)
X_0[3:7] = X_0[3:7] / np.linalg.norm(X_0[3:7])  # normalize the quaternion
# X_0[3:7] = np.array([1, 0, 0, 0])
X_0[7:10] = np.array([0, 2, 0])
X_0[10:13] = np.array([0, -1, 1])

X_hist = np.zeros((N, n_a))  # state vector for each timestep
Fx_hist = np.zeros((N, n_c))  # array of corner Fx for each timestep
Fy_hist = np.zeros((N, n_c))  # array of corner Fy for each timestep
Fz_hist = np.zeros((N, n_c))  # array of corner Fz for each timestep
s1_hist = np.zeros((N, n_c))  # array of slack var 1 values for each timestep
s2_hist = np.zeros((N, n_c))  # array of slack var 2 values for each timestep
lam_hist = np.zeros((N, n_c))  # array of lambda values for each timestep
energy_hist = np.zeros(N)

X_hist[0, :] = X_0
for k in tqdm(range(N - 1)):
    sol = solver(lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, p=X_hist[k, :])
    X_hist[k + 1, :] = np.reshape(sol["x"][0:n_a], (-1,))
    Fx_hist[k] = np.reshape(sol["x"][n_a : n_a + n_c], (-1,))
    Fy_hist[k] = np.reshape(sol["x"][n_a + n_c : n_a + n_c * 2], (-1,))
    Fz_hist[k] = np.reshape(sol["x"][n_a + n_c * 2 : n_a + n_c * 3], (-1,))
    s1_hist[k] = np.reshape(sol["x"][n_a + n_c * 3 : n_a + n_c * 4], (-1,))
    s2_hist[k] = np.reshape(sol["x"][n_a + n_c * 4 : n_a + n_c * 5], (-1,))
    lam_hist[k] = np.reshape(sol["x"][n_a + n_c * 5 :], (-1,))


name = "cube_3d_confr_tstep"
hists = {
    "x (m)": X_hist[:, 0],
    "y (m)": X_hist[:, 1],
    "z (m)": X_hist[:, 2],
    # "vx (m/s)": X_hist[:, 7],
    # "vy (m/s)": X_hist[:, 8],
    # "vz (rad/s)": X_hist[:, 9],
    # "ωx (rad/s)": X_hist[:, 10],
    # "ωy (rad/s)": X_hist[:, 11],
    # "ωz (rad/s)": X_hist[:, 12],
    "Fz_c (N)": Fz_hist,
    "s1": s1_hist,
    "s2": s2_hist,
    "lam": lam_hist,
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
