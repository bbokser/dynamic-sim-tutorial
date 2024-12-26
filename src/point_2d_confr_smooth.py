import numpy as np
import plotting

n_x = 4  # length of state vector
n_u = 2  # length of control vector
m = 10  # mass of the rocket in kg
A = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]])

B = np.array([[0, 0], [0, 0], [1 / m, 0], [0, 1 / m]])

G = np.array([[0, 0, 0, -9.81]]).T
dt = 0.001  # timestep size
mu = 0.1  # coefficient of friction


def get_forces(X):
    z = X[1]
    dx = X[2]
    dz = X[3]
    c = -0.01  # inflection point
    phi = np.clip(
        z, a_min=c + 0.005, a_max=np.inf
    )  # signed distance. clip to just above inflection point
    distance_fn = 1 / (-c + phi) ** 2  # y = 1/x^2 relation
    F_spring = 0.02 * distance_fn  # spring constant inversely related to position
    F_damper = -0.02 * dz * distance_fn  # damper constant inversely related to position
    Fz = F_spring + F_damper
    Fx = -mu * Fz * np.sign(dx)
    a_x = Fx / m
    # don't let friction change the object's direction--that's not possible
    a_x = np.clip(a_x * dt, -dx, 0) / dt
    Fx = a_x * m  # update Fx
    return Fx, Fz


def dynamics_ct(X, U):
    dX = A @ X + B @ U + G.flatten()
    return dX


def integrator_euler(dyn_ct, xk, uk):
    X_next = xk + dt * dyn_ct(xk, uk)
    return X_next


N = 1500  # number of timesteps
X_hist = np.zeros((N, n_x))  # array of state vectors for each timestep
Fx_hist = np.zeros(N)  # array of x GRF forces for each timestep
Fz_hist = np.zeros(N)  # array of z GRF forces for each timestep
X_hist[0, :] = np.array([[0, 1, 1, 0]])
U_hist = np.zeros((N - 1, n_u))  # array of control vectors for each timestep

for k in range(N - 1):
    Fx_hist[k], Fz_hist[k] = get_forces(X_hist[k, :])  # get spring-damper force
    U_hist[k, 0] += Fx_hist[k]  # add friction force to x component of control vector
    U_hist[k, 1] += Fz_hist[k]  # add grf to z component of control vector
    X_hist[k + 1, :] = integrator_euler(dynamics_ct, X_hist[k, :], U_hist[k, :])


name = "2d_confr_smooth"
hists = {
    "x (m)": X_hist[:, 0],
    "z (m)": X_hist[:, 1],
    "dx (m)": X_hist[:, 2],
    "dz (m)": X_hist[:, 3],
    "Fx (N)": Fx_hist,
    "Fz (N)": Fz_hist,
}
plotting.plot_2d_hist(hists, N, name)

# generate animation
plotting.animate(x_hist=X_hist[:, 0], z_hist=X_hist[:, 1], dt=dt, name=name)
