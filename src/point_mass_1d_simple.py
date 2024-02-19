import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt

n_x = 2  # length of state vector
n_u = 1  # length of control vector
m = 10  # mass of the rocket in kg
A = np.array([[0, 1],
              [0, 0]])
B = np.array([[0],
              [1/m]])
G = np.array([[0], [-9.81]])
dt = 0.001  # timestep size

def dynamics_dt(X, U):
    # Solve for X_next by matrix exponential method
    ABG = np.vstack((np.hstack((A, B, G)), np.zeros((n_u + 1, n_x + n_u + 1))))
    M = expm(ABG * dt)
    Ad = M[0:n_x, 0:n_x]
    Bd = M[0:n_x, n_x:n_x + n_u]
    Gd = M[0:n_x, n_x + n_u:]   
    X_next = Ad @ X + Bd @ U + Gd.flatten()
    return X_next

N = 5000  # number of timesteps
X_hist = np.zeros((N, n_x))  # array of state vectors for each timestep
X_hist[0, :] = np.array([[1, 0]])
U_hist = np.zeros((N-1, n_u)) # array of control vectors for each timestep
U_hist[500:750, :] = 1000

for k in range(N-1):
    X_hist[k+1, :] = dynamics_dt(X_hist[k, :], U_hist[k, :])

plt.plot(range(N), X_hist[:, 0], label='expm')
plt.title('Body Position vs Time')
plt.ylabel("z (m)")
plt.xlabel("timesteps")
# plt.legend()
plt.show()
