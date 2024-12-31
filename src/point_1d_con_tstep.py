import numpy as np
import plotting
from scipy.linalg import expm
import casadi as cs

import plotting

n_a = 2  # length of state vector
n_u = 1  # length of control vector
m = 10  # mass of the rocket in kg
g = 9.81  # gravitational constant
A = np.array([[0, 1], [0, 0]])
B = np.array([[0], [1 / m]])
G = np.array([[0], [-g]])
dt = 0.001  # timestep size


# # Discretize by matrix exponential method
# ABG = np.vstack((np.hstack((A, B, G)), np.zeros((n_a, n_a + n_u + 1))))
# M = expm(ABG * dt)
# Ad = M[0:n_a, 0:n_a]
# Bd = M[0:n_a, n_a : n_a + n_u]
# Gd = M[0:n_a, n_a + n_u :]
def dynamics_ct(X, U):
    dX = A @ X + B @ U + G.flatten()
    return dX


def integrator_euler(dyn_ct, xk, uk):
    X_next = xk + dt * dyn_ct(xk, uk)
    return X_next


X_0 = np.array([1, 0])
N = 1000
X_hist = np.zeros((N, n_a))  # array of state vectors for each timestep
F_hist = np.zeros(N)  # array of z GRF forces for each timestep
s_hist = np.zeros(N)  # array of slack var values for each timestep
X_hist[0, :] = X_0
U_hist = np.zeros((N - 1, n_u))  # array of control vectors for each timestep

# initialize casadi variables
Xk1 = cs.SX.sym("Xk1", n_a)  # X(k+1), state at next timestep
F = cs.SX.sym("F", n_u)  # force
s = cs.SX.sym("s", 1)  # slack variable
X = cs.SX.sym("X", n_a)  # state
U = cs.SX.sym("U", n_u)  # controls

zk1 = Xk1[0]  # vert pos
dzk1 = Xk1[1]  # vertical vel

obj = s**2

constr = []  # init constraints
# constr = cs.vertcat(constr, cs.SX(Ad @ X + Bd @ U + Bd @ F + Gd - Xk1))
constr = cs.vertcat(constr, cs.SX(integrator_euler(dynamics_ct, X, U) - Xk1))

# relaxed complementarity aka compl. slackness
constr = cs.vertcat(constr, cs.SX(s - F * zk1))  # ground penetration

opt_variables = cs.vertcat(Xk1, F, s)
# parameters = X
parameters = cs.vertcat(X, U)
lcp = {"x": opt_variables, "p": parameters, "f": obj, "g": constr}
opts = {
    "print_time": 0,
    "ipopt.print_level": 0,
    "ipopt.tol": 1e-6,
    "ipopt.max_iter": 500,
}
solver = cs.nlpsol("S", "ipopt", lcp, opts)

n_var = np.shape(opt_variables)[0]
n_par = np.shape(parameters)[0]
n_g = np.shape(constr)[0]

# variable bounds
ubx = [1e10] * n_var
lbx = [-1e10] * n_var
lbx[0] = 0  # set z positive only
lbx[n_a] = 0  # set F positive only
lbx[-1] = 0  # set slack variable >= 0

# constraint bounds
ubg = [0] * n_g
ubg[-1] = 1e10  # set relaxed complementarity >= 0
lbg = [0] * n_g

# run the sim
p_values = np.zeros(n_par)
x0_values = np.zeros(n_var)
for k in range(N - 1):
    print("timestep = ", k)
    p_values[:n_a] = X_hist[k, :]
    p_values[n_a:] = U_hist[k, :]
    x0_values[:n_a] = X_hist[k, :]
    sol = solver(lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, p=p_values, x0=x0_values)
    X_hist[k + 1, :] = np.reshape(sol["x"][0:n_a], (-1,))
    F_hist[k] = sol["x"][n_a]
    s_hist[k] = sol["x"][n_a + n_u]

pos_hist = X_hist[:, 0]
vel_hist = X_hist[:, 1]

# plotting stuff
name = "1d_con_tstep"
hists = {
    "z (m)": pos_hist,
    "dz (m)": vel_hist,
    "Fz (N)": F_hist,
    "slack var": s_hist,
}
plotting.plot_hist(hists, name)

# generate animation
plotting.animate(
    x_hist=np.zeros(N),
    z_hist=pos_hist,
    dt=dt,
    name=name,
    xlim=[-1, 1],
)
