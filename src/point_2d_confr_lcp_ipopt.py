import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
import casadi as cs

import plotting


def smoothsqrt(x):
    ϵ = 1e-6
    return np.sqrt(x + ϵ * ϵ) - ϵ


n_a = 4  # length of state vector
n_u = 2  # length of control vector

m = 10  # mass of the rocket in kg
g = 9.81  # gravitational constant
A = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]])
B = np.array([[0, 0], [0, 0], [1 / m, 0], [0, 1 / m]])
G = np.array([[0, 0, 0, -9.81]]).T
dt = 0.002  # timestep size
mu = 0.5  # coefficient of friction

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

N = 500  # number of timesteps 5000
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
s1 = cs.SX.sym("s1", N - 1)  # slack variable 1
s2 = cs.SX.sym("s2", N - 1)  # slack variable 2
lamg = cs.SX.sym("lamg", N - 1)  # lagrange mult for ground vel

Q = np.eye(N - 1)
n_t = n_a + n_u  # length of state + control vector
Fx = x[0::n_t]  # friction force
Fz = x[1::n_t]  # grf
phi = x[3::n_t]  # signed distance
dx = x[4::n_t]  # horizontal vel

obj = s1.T @ Q @ s1 + s2.T @ Q @ s2

constr = []  # init constraints
# dynamics
constr = cs.vertcat(constr, cs.SX(C_eq @ x))  # Ax + Bu + G
primal_friction = mu * Fz - smoothsqrt(Fx * Fx)  # uN = Ff
constr = cs.vertcat(constr, cs.SX(primal_friction))  # friction cone
lamg_def = dx - lamg * Fz / smoothsqrt(Fz * Fz)  # tang. gnd vel
constr = cs.vertcat(constr, cs.SX(lamg_def))
# relaxed complementarity
constr = cs.vertcat(constr, cs.SX(s1 - Fz * phi))  # ground penetration
constr = cs.vertcat(constr, cs.SX(s2 - lamg * primal_friction))  # friction
opt_variables = cs.vertcat(x, s1, s2, lamg)
lcp = {"x": opt_variables, "f": obj, "g": constr}
solver = cs.nlpsol("S", "ipopt", lcp)

n_var = np.shape(opt_variables)[0]
# variable bounds
ubx = [1e10] * n_var
lbx = np.append(
    [0] * n_x, [-1e10] * (n_var - n_x)
)  # include lower limit for signed distance
lbx[2:n_x:n_t] = [-1e10 for i in range(N - 1)]  # set x limits
lbx[4:n_x:n_t] = [-1e10 for i in range(N - 1)]  # set dx limits
lbx[5:n_x:n_t] = [-1e10 for i in range(N - 1)]  # set dz limits

n_g = np.shape(constr)[0]
n_f = np.shape(primal_friction)[0]
# constraint bounds
ubg = np.append(d_eq, [1e10] * n_f)
ubg = np.append(ubg, [0] * (n_g - n_eq - n_f))
lbg = np.append(d_eq, [0] * (n_g - n_eq))

sol = solver(lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)

x_sol = sol["x"][:n_x]
Fx_hist = np.array(x_sol[0::n_t])
Fz_hist = np.array(x_sol[1::n_t])
x_hist = np.array(x_sol[2::n_t])
z_hist = np.array(x_sol[3::n_t])
dx_hist = np.array(x_sol[4::n_t])
dz_hist = np.array(x_sol[5::n_t])

# plot w.r.t. time
fig, axs = plt.subplots(2, sharex="all")
fig.suptitle("Body Position vs Time")
plt.xlabel("timesteps")
axs[0].plot(range(N - 1), z_hist)
axs[0].set_ylabel("z (m)")
axs[1].plot(range(N - 1), Fz_hist)
axs[1].set_ylabel("z GRF (N)")
plt.show()

# plot in cartesian coordinatesS
plt.plot(x_hist, z_hist)
plt.title("Body Position in the XZ plane")
plt.xlabel("x (m)")
plt.ylabel("z (m)")
plt.show()

# generate animation
plotting.animate(x_hist=x_hist, z_hist=z_hist, dt=dt, name="2d_confr_lcp")
