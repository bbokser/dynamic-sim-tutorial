import numpy as np
import matplotlib.pyplot as plt

n_x = 2  # length of state vector
n_u = 1  # length of control vector
m = 10  # mass of the rocket in kg
A = np.array([[0, 1],
              [0, 0]])
B = np.array([[0],
              [1/m]])
G = np.array([[0], 
              [-9.81]])
dt = 0.001  # timestep size

def grf(X):
    z = X[0]
    dz = X[1]
    c = -0.01  # inflection point
    phi = np.clip(z, a_min=c+0.005, a_max=np.inf)  # signed distance. clip to just above inflection point
    distance_fn = 1 / (-c+phi)**2  # y = 1/x^2 relation
    F_spring = 0.01 * distance_fn  # spring constant inversely related to position
    F_damper = -0.01 * dz * distance_fn  # damper constant inversely related to position
    grf = F_spring + F_damper
    return grf

def dynamics_ct(X, U):
    dX = A @ X + B @ U + G.flatten()
    return dX

def integrator_euler(dyn_ct, xk, uk):
    X_next = xk + dt * dyn_ct(xk, uk)
    return X_next

N = 5000  # number of timesteps
X_hist = np.zeros((N, n_x))  # array of state vectors for each timestep
F_hist = np.zeros(N)  # array of state vectors for each timestep
X_hist[0, :] = np.array([[1, 0]])
U_hist = np.zeros((N-1, n_u)) # array of control vectors for each timestep

for k in range(N-1):
    F_hist[k] = grf(X_hist[k, :])  # get spring-damper force
    U_hist[k, 0] += F_hist[k]  # add grf to control vector
    X_hist[k+1, :] = integrator_euler(dynamics_ct, X_hist[k, :], U_hist[k, :])

fig, axs = plt.subplots(2, sharex='all')
fig.suptitle('Body Position vs Time')
plt.xlabel('timesteps')
axs[0].plot(range(N), X_hist[:, 0])
axs[0].set_ylabel('z (m)')
axs[1].plot(range(N), F_hist)
axs[1].set_ylabel('GRF (N)')
plt.show()

