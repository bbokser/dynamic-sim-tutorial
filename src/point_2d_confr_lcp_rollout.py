import numpy as np
from scipy.linalg import expm
import casadi as cs

import plotting

ϵ = 1e-6


def smoothsqrt(x):
    return np.sqrt(x + ϵ * ϵ) - ϵ


n_a = 4  # length of state vector
n_u = 2  # length of control vector

m = 10  # mass of the rocket in kg
g = 9.81  # gravitational constant
A = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]])
B = np.array([[0, 0], [0, 0], [1 / m, 0], [0, 1 / m]])
G = np.array([[0, 0, 0, -9.81]]).T
dt = 0.001  # timestep size
mu = 0.1  # coefficient of friction

# Discretize by matrix exponential method
ABG_raw = np.hstack((A, B, G))
ABG_rows = np.shape(ABG_raw)[0]
ABG_cols = np.shape(ABG_raw)[1]
ABG = np.vstack((ABG_raw, np.zeros((ABG_cols - ABG_rows, ABG_cols))))

M = expm(ABG * dt)
Ad = M[0:n_a, 0:n_a]
Bd = M[0:n_a, n_a : n_a + n_u]
Gd = M[0:n_a, n_a + n_u :]

X_0 = np.array([[0, 1, 1, 0]])

N = 1500  # number of timesteps
n_x = (N - 1) * (n_a + n_u)  # number of states in x

ABI = np.hstack((Ad, Bd, -np.eye(n_a)))
ABI_rows = np.shape(ABI)[0]
ABI_cols = np.shape(ABI)[1]

k = 0
n_eq = (N - 1) * ABI_rows  # size of equality constraint

C_eq = np.zeros((n_eq, n_x + n_a))
d_eq = np.zeros((n_eq, 1))

for k in range(N - 1):
    C_eq[
        k * ABI_rows : k * ABI_rows + ABI_rows,
        k * (ABI_cols - n_a) : k * (ABI_cols - n_a) + ABI_cols,
    ] = ABI
    d_eq[k * ABI_rows : k * ABI_rows + n_a, :] = (
        -Gd
    )  # add gravity vector to equality RHS
C_eq = np.delete(C_eq, np.s_[0:n_a], axis=1)  # remove first two columns

d_eq[0:n_a] += np.reshape(-Ad @ X_0.T, (-1, 1))

# X = | Fx1 |  friction force
#     | Fz1 |  GRF or normal force
#     | x2  |  x position
#     | z2  |  z position, aka signed distance
#     | dx2 |  x vel
#     | dz2 |  z vel
#     | ... |
#     | xN  |
#     | zN  |
#     | dxN |
#     | dzN |

x = cs.SX.sym("x", n_x)  # combination of states and controls (grfs)
s1 = cs.SX.sym("s1", N - 2)  # slack variable 1
s2 = cs.SX.sym("s2", N - 1)  # slack variable 2
lam = cs.SX.sym("lam", N - 1)  # lagrange mult for ground vel

Q1 = np.eye(N - 2)
Q2 = np.eye(N - 1)
n_t = n_a + n_u  # length of state + control vector
Fx = x[0::n_t]  # friction force
Fz = x[1::n_t]  # grf
z = x[3::n_t]  # signed distance
dx = x[4::n_t]  # horizontal vel
zk1 = x[7::n_t]  # signed distance, next timestep
obj = s1.T @ Q1 @ s1 + s2.T @ Q2 @ s2

constr = []  # init constraints
# dynamics
constr = cs.vertcat(constr, cs.SX(C_eq @ x))  # Ax + Bu + G
lam_def = dx + lam * Fx / (smoothsqrt(Fx * Fx) + ϵ)  # tang. gnd vel
constr = cs.vertcat(constr, cs.SX(lam_def))

primal_friction = mu * Fz - smoothsqrt(Fx * Fx)  # uN = Ff
constr = cs.vertcat(constr, cs.SX(primal_friction))  # friction cone

# relaxed complementarity
constr = cs.vertcat(constr, cs.SX(s1 - Fz[:-1] * zk1))  # ground penetration
constr = cs.vertcat(constr, cs.SX(s2 - lam * primal_friction))  # friction
opt_variables = cs.vertcat(x, s1, s2, lam)
lcp = {"x": opt_variables, "f": obj, "g": constr}
opts = {
    "print_time": 0,
    "ipopt.print_level": 0,
    "ipopt.tol": ϵ,
    "ipopt.max_iter": 1000,
}
solver = cs.nlpsol("S", "ipopt", lcp, opts)

n_var = np.shape(opt_variables)[0]
# variable bounds
ubx = [1e10] * n_var
lbx = [0] * n_var  # include lower limit for signed distance

lbx[0:n_x:n_t] = [-1e10 for i in range(N - 1)]  # set Fx limits
lbx[2:n_x:n_t] = [-1e10 for i in range(N - 1)]  # set x limits
lbx[4:n_x:n_t] = [-1e10 for i in range(N - 1)]  # set dx limits
lbx[5:n_x:n_t] = [-1e10 for i in range(N - 1)]  # set dz limits

n_g = np.shape(constr)[0]
n_deq = np.shape(d_eq)[0]
n_lam = np.shape(lam_def)[0]
n_f = np.shape(primal_friction)[0]
# constraint bounds

ubg = [1e10] * n_g
ubg[:n_deq] = d_eq
ubg[n_deq : n_deq + n_lam] = [0 for i in range(n_lam)]

lbg = np.append(d_eq, [0] * (n_g - n_eq))

sol = solver(lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)

x_sol = sol["x"][:n_x]
Fx_hist = np.array(x_sol[0::n_t])
Fz_hist = np.array(x_sol[1::n_t])
x_hist = np.array(x_sol[2::n_t])
z_hist = np.array(x_sol[3::n_t])
dx_hist = np.array(x_sol[4::n_t])
dz_hist = np.array(x_sol[5::n_t])
s1_hist = np.array(x_sol[n_a + 1 :: n_t])
s2_hist = np.array(x_sol[n_a + 1 :: n_t])
lam_hist = np.array(x_sol[n_a + 1 :: n_t])

name = "2d_confr_lcp_rollout"
hists = {
    "x (m)": x_hist,
    "z (m)": z_hist,
    "dx (m)": dx_hist,
    "dz (m)": dz_hist,
    "Fx (N)": Fx_hist,
    "Fz (N)": Fz_hist,
    "slack var1": s1_hist,
    "slack var2": s2_hist,
    "lambda": lam_hist,
}
plotting.plot_hist(hists, name)
# generate animation
plotting.animate(x_hist=x_hist, z_hist=z_hist, dt=dt, name=name)
