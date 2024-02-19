import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
from copy import copy

n_x = 2  # length of state vector
n_u = 1  # length of control vector
m = 10  # mass of the rocket in kg
A = np.array([[0, 1],
              [0, 0]])
B = np.array([[0],
              [1/m]])
G = np.array([[0], [-9.81]])
dt = 0.1  # timestep size

# Discretize by matrix exponential method
ABG = np.vstack((np.hstack((A, B, G)), np.zeros((n_u + 1, n_x + n_u + 1))))
M = expm(ABG * dt)
Ad = M[0:n_x, 0:n_x]
Bd = M[0:n_x, n_x:n_x + n_u]
Gd = M[0:n_x, n_x + n_u:]  

def dynamics_dt(X, U):  
    X_next = Ad @ X + Bd @ U + Gd.flatten()
    return X_next

def dynamics_ct(X, U):
    dX = A @ X + B @ U + G.flatten()
    return dX

def integrator_euler(dyn_ct, xk, uk):
    # Euler integrator solves for X_next
    X_next = xk + dt * dyn_ct(xk, uk)
    return X_next

def integrator_rk4(dyn_ct, xk, uk):
    # RK4 integrator solves for X_next
    f1 = dyn_ct(xk, uk)
    f2 = dyn_ct(xk + 0.5 * dt * f1, uk)
    f3 = dyn_ct(xk + 0.5 * dt * f2, uk)
    f4 = dyn_ct(xk + dt * f3, uk)
    return xk + (dt / 6.0) * (f1 + 2 * f2 + 2 * f3 + f4)

# X_0 = np.array([[1, 0]])
# U_0 = np.zeros(n_u)

N = 10  # number of timesteps
X_hist_expm = np.zeros((N, n_x))  # array of state vectors for each timestep
X_hist_expm[0, :] = np.array([[1, 0]])
X_hist_euler = copy(X_hist_expm)
X_hist_rk4 = copy(X_hist_expm)
U_hist = np.zeros((N-1, n_u)) # array of control vectors for each timestep

for k in range(N-1):
    X_hist_rk4[k+1, :] = integrator_rk4(dynamics_ct, X_hist_rk4[k, :], U_hist[k, :])
    X_hist_euler[k+1, :] = integrator_euler(dynamics_ct, X_hist_euler[k, :], U_hist[k, :])
    X_hist_expm[k+1, :] = dynamics_dt(X_hist_expm[k, :], U_hist[k, :])

plt.plot(range(N), X_hist_expm[:, 0], label='expm', linewidth=5)
plt.plot(range(N), X_hist_euler[:, 0], label='euler')
plt.plot(range(N), X_hist_rk4[:, 0], label='rk4')
plt.title('Body Position vs Time')
plt.ylabel("z (m)")
plt.xlabel("timesteps")
plt.legend()
plt.show()
