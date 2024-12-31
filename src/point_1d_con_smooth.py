import numpy as np
import plotting


def get_grf(X: np.ndarray) -> float:
    z = X[0]
    dz = X[1]
    k = 0.01  # spring constant
    b = 0.1  # damping constant
    amp = 1500  # desired max force
    c = amp * 0.5 / k
    distance_fn = -c * np.tanh(z * 100) + c
    F_spring = k * distance_fn
    F_damper = -b * dz * distance_fn
    grf = F_spring + F_damper
    return grf


def main():
    n_x = 2  # length of state vector
    n_u = 1  # length of control vector
    m = 10  # mass of the rocket in kg
    A = np.array([[0, 1], [0, 0]])
    B = np.array([[0], [1 / m]])
    G = np.array([[0], [-9.81]])
    dt = 0.001  # timestep size

    def dynamics_ct(X, U):
        dX = A @ X + B @ U + G.flatten()
        return dX

    def integrator_euler(dyn_ct, xk, uk):
        X_next = xk + dt * dyn_ct(xk, uk)
        return X_next

    N = 1000  # number of timesteps
    X_hist = np.zeros((N, n_x))  # array of state vectors for each timestep
    F_hist = np.zeros((N, n_u))  # array of state vectors for each timestep
    X_hist[0, :] = np.array([[1, 0]])

    for k in range(N - 1):
        F_hist[k, :] = get_grf(X_hist[k, :])  # get spring-damper force
        X_hist[k + 1, :] = integrator_euler(dynamics_ct, X_hist[k, :], F_hist[k, :])

    # plotting stuff
    name = "1d_con_smooth"
    hists = {
        "z (m)": X_hist[:, 0],
        "dz (m)": X_hist[:, 1],
        "Fz (N)": F_hist,
    }
    plotting.plot_hist(hists, name, ylim=[0, 1500])

    # generate animation
    plotting.animate(
        x_hist=np.zeros(N),
        z_hist=X_hist[:, 0],
        dt=dt,
        name=name,
        xlim=[-1, 1],
    )


if __name__ == "__main__":
    main()
