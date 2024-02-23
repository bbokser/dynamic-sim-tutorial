import numpy as np
import matplotlib.pyplot as plt

n_x = 2  # length of state vector
n_u = 1  # length of control vector
m = 10  # mass of the rocket in kg
k = 10000  # spring constant
b = 100  # damper constant
A = np.array([[0, 1],
              [0, 0]])
A_contact = np.array([[0, 1],
                      [-k/m, -b/m]])
B = np.array([[0],
              [1/m]])
G = np.array([[0], 
              [-9.81]])
dt = 0.001  # timestep size

def dynamics_ct(X, U):
    dX = A @ X + B @ U + G.flatten()
    return dX

def dynamics_ct_contact(X, U):
    dX = A_contact @ X + B @ U + G.flatten()
    return dX

def integrator_euler(dyn_ct, xk, uk):
    X_next = xk + dt * dyn_ct(xk, uk)
    return X_next

N = 5000  # number of timesteps
X_hist = np.zeros((N, n_x))  # array of state vectors for each timestep
X_hist[0, :] = np.array([[1, 0]])
U_hist = np.zeros((N-1, n_u)) # array of control vectors for each timestep

for k in range(N-1):
    if X_hist[k, 0] < 0:  # guard function
        X_hist[k+1, :] = integrator_euler(dynamics_ct_contact, X_hist[k, :], U_hist[k, :])
    else:
        X_hist[k+1, :] = integrator_euler(dynamics_ct, X_hist[k, :], U_hist[k, :])

plt.plot(range(N), X_hist[:, 0])
plt.title('Body Position vs Time')
plt.ylabel("z (m)")
plt.xlabel("timesteps")
plt.show()
