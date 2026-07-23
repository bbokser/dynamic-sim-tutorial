import numpy as np
from tqdm import tqdm
import casadi as cs
from collections.abc import Callable
from transforms import H, Lq
from transforms_cs import Aq_cs, Lq_cs
import plotting
from cube_3d_floating import (
    rk4_normalized,
    kin_corners,
    animate_cube,
    plot_energy,
    G,
    MASS,
    INERTIA,
    I_INV,
    DT,
    R_C_B,
)


def dynamics_falling_ct(X: np.ndarray, F: np.ndarray) -> np.ndarray:
    """
    Continuous-time SE(3) nonlinear dynamics
    Subject to gravity but not disturbance forces

    :param X: state vector
    :param F: forces (ignored)
    """
    # Unpack state vector
    Q = X[3:7]  # B to W
    v_w = X[7:10]  # W frame
    ω_b = X[10:13]  # B frame
    F_w = np.array([0, 0, -G]) * MASS  # force in W frame

    dr = v_w
    dq = 0.5 * Lq(Q) @ H @ ω_b
    dv = 1 / MASS * F_w
    dω = np.zeros(3)
    dX = np.hstack((dr, dq, dv, dω))
    return dX


def kin_corners_cs(X: cs.SX) -> cs.SX:
    """
    Get world frame locations of the 8 corners of the cube
    (CaSaDi version)

    :param X: state vector
    """
    r_c = cs.SX(8, 3)
    ones_nc = cs.SX.ones(8, 1)
    r_w = X[0:3]  # W frame
    Q = X[3:7]  # B to W
    A = Aq_cs(Q)  # rotation matrix
    r_c = (A @ R_C_B.T).T + ones_nc @ r_w.T
    return r_c


def dynamics_con_ct(X: cs.SX, F: cs.SX) -> cs.SX:
    """
    Continuous-time SE(3) nonlinear dynamics
    Subject to gravity and vertical collision forces

    :param X: state vector
    :param F: forces (8x1), scalar value per corner
    """
    # r_w = X[0:3]  # W frame
    Q = X[3:7]  # B to W
    v_w = X[7:10]  # W frame
    ω_b = X[10:13]  # B frame
    F_w = cs.SX(3, 1)  # force in W frame
    tau_b = cs.SX(3, 1)  # torque in B frame

    # rotation matrix, body to world frame
    A = Aq_cs(Q)
    # iterate through corners to calculate torque acting on body
    F_c_w = cs.SX(8, 3)
    for i in range(8):
        F_c_w[i, 2] = F[i]
        F_w += F_c_w[i, :].T
        # add body frame torque due to body frame force
        tau_b += cs.cross(R_C_B[i, :], A.T @ F_c_w[i, :].T)

    F_w += np.array([0, 0, -9.81]) * MASS  # gravity

    dr = v_w
    dq = 0.5 * Lq_cs(Q) @ H @ ω_b
    dv = 1 / MASS * F_w
    dω = I_INV @ (tau_b - cs.cross(ω_b, INERTIA @ ω_b))
    dX = cs.vertcat(dr, dq, dv, dω)
    return dX


def euler_semi_implicit(
    dynamics: Callable,
    X_k: cs.SX,
    U_k: cs.SX,
    X_k1: cs.SX,
) -> cs.SX:
    """
    Semi-Implicit Euler Integrator

    :param dynamics: dynamics function
    :param X_k: state vector at step k
    :param U_k: control vector at step k
    :param X_k1: state vector at step k+1
    """
    X_k_semi = cs.SX.zeros(13)
    X_k_semi[:7] = X_k[:7]
    X_k_semi[7:] = X_k1[7:]
    X_n = X_k + DT * dynamics(X_k_semi, U_k)
    X_n[3:7] = X_n[3:7] / cs.norm_2(X_n[3:7])  # normalize the quaternion term
    return X_n


def main():
    n_a = 13  # length of state vector
    n_c = 8  # number of contact points on cube
    # initialize casadi variables
    Xk1 = cs.SX.sym("Xk1", n_a)  # X(k+1), state at next timestep
    F = cs.SX.sym("F", n_c)  # force at each corner
    s = cs.SX.sym("s", n_c)  # slack variable
    X = cs.SX.sym("X", n_a)  # X(k), state

    obj = s.T @ s

    constr = []  # init constraints
    # constr = cs.vertcat(constr, rk4_normalized(dynamics_ct, X, F) - Xk1)
    constr = cs.vertcat(constr, euler_semi_implicit(dynamics_con_ct, X, F, Xk1) - Xk1)

    # stay above the ground
    z_c_w = kin_corners_cs(Xk1)[:, 2]  # corner heights
    constr = cs.vertcat(constr, z_c_w)

    # relaxed complementarity aka compl. slackness
    constr = cs.vertcat(constr, s - F * z_c_w)  # ground penetration

    opt_variables = cs.vertcat(Xk1, F, s)
    lcp = {"x": opt_variables, "p": X, "f": obj, "g": constr}
    opts = {
        "print_time": 0,
        "ipopt.print_level": 0,
        "ipopt.tol": 1e-8,
        "ipopt.max_iter": 1500,
    }
    solver = cs.nlpsol("S", "ipopt", lcp, opts)

    n_var = np.shape(opt_variables)[0]
    n_g = np.shape(constr)[0]

    # variable bounds
    ubx = [1e10] * n_var
    lbx = [0] * n_var
    lbx[:n_a] = [-1e10] * n_a  # state can be negative

    # constraint bounds
    ubg = [0] * n_g
    ubg[n_a : n_a + n_c] = [1e10] * n_c  # set z_c >= 0
    ubg[n_a + n_c :] = [1e10] * n_c  # set relaxed complementarity >= 0
    lbg = [0] * n_g

    # initialize simulation variables
    N = 2000  # number of timesteps
    X_0 = np.zeros(n_a)
    X_0[:3] = np.array([0, 0, 3.0])
    # create random quaternion
    X_0[3:7] = np.random.rand(4)
    # normalize quaternion
    X_0[3:7] = X_0[3:7] / np.linalg.norm(X_0[3:7])
    X_0[7:10] = np.array([0, 2, 0])
    X_0[10:13] = np.array([0, -1, 1])

    X_hist = np.zeros((N, n_a))  # array of state vectors for each timestep
    F_hist = np.zeros((N, n_c))  # array of GRF for each timestep
    s_hist = np.zeros((N, n_c))  # array of slack var values for each timestep

    X_hist[0, :] = X_0
    for k in tqdm(range(N - 1)):
        X_hist[k + 1, :] = rk4_normalized(
            dynamics_falling_ct, X_hist[k, :], np.zeros(8)
        )
        if (kin_corners(X_hist[k + 1, :])[:, 2] <= 0).any() or X_hist[k + 1, 2] <= 1:
            sol = solver(lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, p=X_hist[k, :])
            X_hist[k + 1, :] = np.reshape(sol["x"][0:n_a], (-1,))
            F_hist[k] = np.reshape(sol["x"][n_a : n_a + n_c], (-1,))
            s_hist[k] = np.reshape(sol["x"][n_a + n_c :], (-1,))

    name = "cube_3d_con_tstep"
    hists = {
        "x (m)": X_hist[:, 0],
        "y (m)": X_hist[:, 1],
        "z (m)": X_hist[:, 2],
        "F_c (N)": F_hist,
        "s": s_hist,
    }
    plotting.plot_hist(hists, name)
    animate_cube(X_hist, name)
    plot_energy(X_hist, name)


if __name__ == "__main__":
    main()
