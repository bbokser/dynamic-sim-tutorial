import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
import cvxpy as cp

import matplotlib.pylab as pylt

n_a = 2  # length of state vector
n_u = 1  # length of control vector
m = 10  # mass of the rocket in kg
g = 9.81  # gravitational constant
A = np.array([[0, 1],
              [0, 0]])
B = np.array([[0],
              [1/m]])
G = np.array([[0], 
              [-g]])
dt = 0.001  # timestep size

# Discretize by matrix exponential method
ABG = np.vstack((np.hstack((A, B, G)), np.zeros((n_a, n_a + n_u + 1))))
M = expm(ABG * dt)
Ad = M[0:n_a, 0:n_a]
Bd = M[0:n_a, n_a:n_a + n_u]
Gd = M[0:n_a, n_a + n_u:]  

X_0 = np.array([0.5, 0])

N = 100  # number of timesteps 5000
n_x  = (N-1) * (n_a + n_u)  # number of states in x

ABI = np.hstack((Ad, Bd, -np.eye(n_a)))
ABI_rows = np.shape(ABI)[0]
ABI_cols = np.shape(ABI)[1]

k = 0
n_eq = (N-1) * ABI_rows  # size of equality constraint

C_eq = np.zeros((n_eq, n_x + n_a))
d_eq = np.zeros((n_eq, 1))

n_z = n_x
C_ineq = np.eye(n_x)
# n_z = n_x * 2
# C_ineq = np.vstack((np.eye(n_x), np.zeros((n_x, n_x))))
d_ineq = np.zeros(n_z)

for k in range(N-1):
   C_eq[k * ABI_rows : k * ABI_rows + ABI_rows, k * (ABI_cols - n_a):k * (ABI_cols-n_a) + ABI_cols] = ABI
   C_ineq[k * (n_a+n_u) + 2, k * (n_a+n_u) + 2] = 0  # multiplier for xdot
   d_eq[k * ABI_rows: k * ABI_rows + n_a, :] = -Gd  # add gravity vector to equality RHS
C_eq = np.delete(C_eq, np.s_[0:n_a], axis=1)  # remove first two columns
d_eq[0:n_a] += np.reshape(-Ad @ X_0, (-1, 1))
# print(np.shape(C_eq))

Q = np.eye(n_x)  # actually combination of Q and R
M_eq = np.linalg.inv(np.hstack((np.vstack((Q, -C_eq)), np.vstack((C_eq.T, np.zeros((n_eq, n_eq)))))))
n_M_eq = np.shape(M_eq)[0]

# print(np.shape(C_ineq))
# print(np.shape(M_eq))

C_ineq_stacked = np.hstack((C_ineq, np.zeros((n_z, n_M_eq - n_x))))
M = C_ineq_stacked @ M_eq @ (C_ineq_stacked.T)
q = -C_ineq_stacked @ M_eq @ np.vstack((np.zeros((n_M_eq - n_eq, 1)), d_eq)) - d_ineq.reshape(-1, 1)
# print(np.shape(C_ineq_stacked))
# print(np.shape(M))
# print(np.shape(q))
z = cp.Variable((n_z, 1))
# pylt.matshow(q)
# pylt.gca().set_aspect('auto')
# pylt.show()

# cost = cp.quad_form(z, Q)
# constr = []  # init constraints
# constr += [C_eq @ z == d_eq]
# constr += [C_ineq @ z >= 0]

# --- #
cost = cp.quad_form(z, M) + q.T @ z  # cost function

constr = []  # init constraints
constr += [M @ z + q >= 0] # dynamics
constr += [z >= 0]
# lam = z[::3]  # grf
# phi = z[1::3]  # signed distance
# constr += [phi[0] == 1]
# constr += [lam >= 0]
# constr += [phi >= 0]

# --- set up solver --- #
problem = cp.Problem(cp.Minimize(cost), constr)
problem.solve(verbose=True)

#grf = U.value[:, 1]
F_hist = z.value[0::3]
pos_hist = z.value[1::3]  # array of state vectors for each timestep
vel_hist = z.value[2::3]

fig, axs = plt.subplots(3, sharex='all')
fig.suptitle('Body Position vs Time')
plt.xlabel('timesteps')
axs[0].plot(range(N-1), pos_hist)
axs[0].set_ylabel('z (m)')
axs[0].set_ylim([-0.5, 1])
axs[1].plot(range(N-1), vel_hist)
axs[1].set_ylabel('z dot (m/s)')
axs[2].plot(range(N-1), F_hist)
axs[2].set_ylabel('z GRF (N)')
plt.show()