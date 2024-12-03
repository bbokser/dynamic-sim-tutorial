import numpy as np
import matplotlib.pyplot as plt
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
dt = 0.002  # timestep size

# Discretize by matrix exponential method
ABG = np.vstack((np.hstack((A, B, G)), np.zeros((n_a, n_a + n_u + 1))))
M = expm(ABG * dt)
Ad = M[0:n_a, 0:n_a]
Bd = M[0:n_a, n_a : n_a + n_u]
Gd = M[0:n_a, n_a + n_u :]

X_0 = np.array([1, 0])

N = 500  # number of timesteps
n_au = n_a + n_u
n_x = (N - 1) * (n_au)  # number of states in x

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
d_eq[0:n_a] += np.reshape(-Ad @ X_0, (-1, 1))

x = cs.SX.sym("x", n_x)  # combination of states and controls (grfs)
s = cs.SX.sym("s", N - 1)  # slack variable
Q = np.eye(N - 1)
lam = x[::n_au]  # grf
phi = x[1::n_au]  # signed distance
obj = s.T @ Q @ s

constr = []  # init constraints
constr = cs.vertcat(constr, cs.SX(C_eq @ x))  # Ax + Bu + G
constr = cs.vertcat(constr, cs.SX(s - lam * phi))  # relaxed complementarity
# constr = cs.vertcat(constr, C_ineq @ x)  # inequality constraint >=
opt_variables = cs.vertcat(x, s)
lcp = {"x": opt_variables, "f": obj, "g": constr}
opts = {
    "print_time": 0,
    "ipopt.print_level": 0,
    # "ipopt.tol": 1e-4,
    # "ipopt.max_iter": 1000,
}
solver = cs.nlpsol("S", "ipopt", lcp, opts)

n_var = np.shape(opt_variables)[0]
ubx = [1e10] * (n_x + N - 1)
lbx = [-1e10] * (n_x + N - 1)
# set F positive only
lbx[0:n_x:n_au] = [0 for i in range(N - 1)]
# set z positive only
lbx[1:n_x:n_au] = [0 for i in range(N - 1)]
# set slack variable >= 0
lbx[n_x:] = [0 for i in range(N - 1)]

# set relaxed complementarity >= 0
ubg = np.append(d_eq, [1e10] * (N - 1))
lbg = np.append(d_eq, [0] * (N - 1))
sol = solver(lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)

x_sol = sol["x"][:n_x]
F_hist = np.array(x_sol[0::n_au])
pos_hist = np.array(x_sol[1::n_au])
vel_hist = np.array(x_sol[2::n_au])

fig, axs = plt.subplots(3, sharex="all")
fig.suptitle("Body Position vs Time")
plt.xlabel("timesteps")
axs[0].plot(range(N - 1), pos_hist)
axs[0].set_ylabel("z (m)")
axs[0].set_ylim([-0.5, 1])
axs[1].plot(range(N - 1), vel_hist)
axs[1].set_ylabel("z dot (m/s)")
axs[2].plot(range(N - 1), F_hist)
axs[2].set_ylabel("z GRF (N)")
plt.show()

plotting.animate(
    x_hist=np.zeros(N), z_hist=pos_hist, dt=dt, name="1d_con_lcp_rollout", xlim=[-1, 1]
)
