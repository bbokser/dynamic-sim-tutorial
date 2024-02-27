import numpy as np
import matplotlib.pyplot as plt

n_x = 2  # length of state vector
n_u = 1  # length of control vector
m = 10  # mass of the rocket in kg
k = 10000  # spring constant
b = 100  # damper constant
A = np.array([[0, 1],
              [0, 0]])
A_spring = np.array([[0, 1],
                     [-k/m, -b/m]])
B = np.array([[0],
              [1/m]])
G = np.array([[0], 
              [-9.81]])
dt = 0.001  # timestep size
e = 0.7  # coefficient of restitution

def dynamics_ct(X, U):
    dX = A @ X + B @ U + G.flatten()
    return dX

def integrator_euler(dyn_ct, xk, uk):
    X_next = xk + dt * dyn_ct(xk, uk)
    return X_next

def dynamics_ct_spring(X, U):
    dX = A_spring @ X + B @ U + G.flatten()
    return dX

def jump_map(X):  #
    X[0] = 0  # reset z position to zero
    v_before = X[1]  # velocity before impact
    v_after = -e * v_before  # reverse velocity and multiply by coefficient of restitution
    a = (v_after - v_before)/dt  # acceleration
    F = m * a  # get ground reaction force
    X[1] = v_after  # velocity after impact
    return X, F

N = 5000  # number of timesteps
X_hist = np.zeros((N, n_x))  # array of state vectors for each timestep
F_hist = np.zeros((N, 1))  # array of state vectors for each timestep
X_hist[0, :] = np.array([[1, 0]])
U_hist = np.zeros((N-1, n_u)) # array of control vectors for each timestep

# uncomment one of these to choose
contact_model = 'jump_map'  
# contact_model = 'spring'

if contact_model == 'jump_map':
    for k in range(N-1):
        if X_hist[k, 0] < 0:  # guard function
            X_hist[k, :], F_hist[k, :] = jump_map(X_hist[k, :])  # dynamics rewrite based on impact
        X_hist[k+1, :] = integrator_euler(dynamics_ct, X_hist[k, :], U_hist[k, :])

elif contact_model == 'spring':
    for k in range(N-1):
        if X_hist[k, 0] < 0:  # guard function
            X_hist[k+1, :] = integrator_euler(dynamics_ct_spring, X_hist[k, :], U_hist[k, :])
        else:
            X_hist[k+1, :] = integrator_euler(dynamics_ct, X_hist[k, :], U_hist[k, :])

fig, axs = plt.subplots(2, sharex='all')
fig.suptitle('Body Position vs Time')
plt.xlabel('timesteps')
axs[0].plot(range(N), X_hist[:, 0])
axs[0].set_ylabel('z (m)')
axs[1].plot(range(N), F_hist)
axs[1].set_ylabel('GRF (N)')
plt.show()
