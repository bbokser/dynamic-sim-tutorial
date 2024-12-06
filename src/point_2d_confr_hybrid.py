import numpy as np
import matplotlib.pyplot as plt

import plotting

n_x = 4  # length of state vector
n_u = 2  # length of control vector
m = 10  # mass of the rocket in kg
A = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]])

B = np.array([[0, 0], [0, 0], [1 / m, 0], [0, 1 / m]])

G = np.array([[0, 0, 0, -9.81]]).T
dt = 0.001  # timestep size
e = 0.0  # coefficient of restitution
mu = 0.1  # coefficient of friction


def dynamics_ct(X, U):
    dX = A @ X + B @ U + G.flatten()
    return dX


def integrator_euler(dyn_ct, xk, uk):
    X_next = xk + dt * dyn_ct(xk, uk)
    return X_next


def jump_map(X):
    X[1] = 0  # reset z position to zero
    dz_before = X[3]  # z velocity before impact
    dz_after = (
        -e * dz_before
    )  # reverse velocity and multiply by coefficient of restitution
    X[3] = dz_after  # z velocity after impact
    return X


N = 5000  # number of timesteps
X_hist = np.zeros((N, n_x))  # array of state vectors for each timestep
Fx_hist = np.zeros(N)  # array of x GRF forces for each timestep
Fz_hist = np.zeros(N)  # array of z GRF forces for each timestep
X_hist[0, :] = np.array([[0, 1, 1, 0]])
U_hist = np.zeros((N - 1, n_u))  # array of control vectors for each timestep

for k in range(N - 1):
    z = X_hist[k, 1]
    dz = X_hist[k, 3]
    # if z position is below ground
    if z < 0:  # guard function
        X_hist[k, :] = jump_map(X_hist[k, :])  # dynamics rewrite based on impact

    # if z position is zero and z vel is zero or negative, you are in ground contact and have friction
    if z == 0 and dz <= 0:
        dz_before = dz  # z velocity before
        dz_after = 0  # desired velocity is zero
        a_z = (dz_after - dz_before) / dt  # acceleration
        Fz = m * a_z  # get required ground reaction force to stay aboveground
        Fz_hist[k] = Fz

        dx = X_hist[k, 2]
        Fx = -mu * Fz_hist[k] * np.sign(dx)
        a_x = Fx / m
        # don't let friction change the object's direction--that's not possible
        a_x = np.clip(a_x * dt, -dx, 0) / dt
        Fx_hist[k] = a_x * m  # update Fx
        U_hist[k, 0] += Fx_hist[k]

    X_hist[k + 1, :] = integrator_euler(dynamics_ct, X_hist[k, :], U_hist[k, :])

# plot w.r.t. time
fig, axs = plt.subplots(2, sharex="all")
fig.suptitle("Body Position vs Time")
plt.xlabel("timesteps")
axs[0].plot(range(N), X_hist[:, 1])
axs[0].set_ylabel("z (m)")
axs[1].plot(range(N), Fz_hist)
axs[1].set_ylabel("z GRF (N)")
plt.show()

plotting.animate(
    x_hist=X_hist[:, 0], z_hist=X_hist[:, 1], dt=dt, name="2d_confr_hybrid"
)
