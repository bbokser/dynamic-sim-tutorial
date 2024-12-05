import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
import casadi as cs

import plotting

ϵ = 1e-6


def smoothsqrt(x):
    return cs.sqrt(x + ϵ * ϵ) - ϵ


n_a = 4  # length of state vector
n_u = 2  # length of control vector

m = 10  # mass of the rocket in kg
g = 9.81  # gravitational constant
A = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]])
B = np.array([[0, 0], [0, 0], [1 / m, 0], [0, 1 / m]])
G = np.array([[0, 0, 0, -9.81]]).T
dt = 0.002  # timestep size
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

N = 500  # number of timesteps
X_hist = np.zeros((N, n_a))  # array of state vectors for each timestep
Fx_hist = np.zeros(N)  # array of x friction forces for each timestep
Fz_hist = np.zeros(N)  # array of z GRF forces for each timestep
X_hist[0, :] = np.array([[0, 1, 1, 0]])
U_hist = np.zeros((N - 1, n_u))  # array of control vectors for each timestep

# initialize casadi variables
Xk1 = cs.SX.sym("Xk1", n_a)  # X(k+1), state at next timestep
F = cs.SX.sym("F", n_u)  # forces
s1 = cs.SX.sym("s1", 1)  # slack variable 1
s2 = cs.SX.sym("s2", 1)  # slack variable 2
lam = cs.SX.sym("lam", 1)  # lagrange mult for ground vel
X = cs.SX.sym("X", n_a)  # state
U = cs.SX.sym("U", n_u)  # controls

x = Xk1[0]  # horz pos
z = Xk1[1]  # vert pos
dx = Xk1[2]  # horizontal vel
dz = Xk1[3]  # vertical vel
Fx = F[0]  # friction force
Fz = F[1]  # grf

obj = s1 + s2

constr = []  # init constraints
# dynamics A*X(k) + B*U(k) + G(k) - X(k+1) = 0
constr = cs.vertcat(constr, cs.SX(Ad @ X + Bd @ U + Bd @ F + Gd - Xk1))

# tang. gnd vel is zero if GRF is zero but is otherwise equal to dx
# max dissipation
constr = cs.vertcat(constr, cs.SX(dx + lam * Fx / (smoothsqrt(Fx * Fx) + ϵ)))

# primal feasibility
primal_friction = mu * Fz - smoothsqrt(Fx * Fx)  # uN = Ff
constr = cs.vertcat(constr, cs.SX(primal_friction))  # friction cone

# relaxed complementarity aka compl. slackness
constr = cs.vertcat(constr, cs.SX(s1 - Fz * z))  # ground penetration
constr = cs.vertcat(constr, cs.SX(s2 - lam * primal_friction))  # friction

opt_variables = cs.vertcat(Xk1, F, s1, s2, lam)
parameters = cs.vertcat(X, U)
lcp = {"x": opt_variables, "p": parameters, "f": obj, "g": constr}
opts = {
    "print_time": 0,
    "ipopt.print_level": 0,
    "ipopt.tol": ϵ,
    "ipopt.max_iter": 1000,
}
solver = cs.nlpsol("S", "ipopt", lcp, opts)

n_var = np.shape(opt_variables)[0]
n_par = np.shape(parameters)[0]
n_g = np.shape(constr)[0]

# variable bounds
ubx = [1e10] * n_var
lbx = [0] * n_var  # dual feasibility
lbx[0] = -1e10  # set x unlimited
lbx[2] = -1e10  # set dx unlimited
lbx[3] = -1e10  # set dz unlimited
lbx[n_a] = -1e10  # set Fx unlimited

# constraint bounds
ubg = [1e10] * n_g
ubg[0:n_a] = np.zeros(n_a)  # set dynamics = 0
ubg[n_a] = 0  # set max dissipation = 0
lbg = [0] * n_g

# run the sim
p_values = np.zeros(n_par)
x0_values = np.zeros(n_var)
for k in range(N - 1):
    print("timestep = ", k)
    p_values[:n_a] = X_hist[k, :]
    p_values[n_a:] = U_hist[k, :]
    # x0_values[:n_a] = X_hist[k, :]
    sol = solver(lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, p=p_values)  # , x0=x0_values)
    X_hist[k + 1, :] = np.reshape(sol["x"][0:n_a], (-1,))
    Fx_hist[k] = sol["x"][n_a]
    Fz_hist[k] = sol["x"][n_a + 1]

x_hist = X_hist[:, 0]
z_hist = X_hist[:, 1]
# plot w.r.t. time
fig, axs = plt.subplots(4, sharex="all")
fig.suptitle("Body Position vs Time")
plt.xlabel("timesteps")
axs[0].plot(range(N), x_hist)
axs[0].set_ylabel("x (m)")
axs[1].plot(range(N), z_hist)
axs[1].set_ylabel("z (m)")
axs[2].plot(range(N), Fx_hist)
axs[2].set_ylabel("x F (N)")
axs[3].plot(range(N), Fz_hist)
axs[3].set_ylabel("z F (N)")
plt.show()

# plot in cartesian coordinatesS
plt.plot(x_hist, z_hist)
plt.title("Body Position in the XZ plane")
plt.xlabel("x (m)")
plt.ylabel("z (m)")
plt.show()

# generate animation
plotting.animate(x_hist=x_hist, z_hist=z_hist, dt=dt, name="2d_confr_lcp_tstepping")
