import numpy as np
from tqdm import tqdm
import casadi as cs
import plotting
from cube_3d_floating import (
    rk4_normalized,
    kin_corners,
    animate_cube,
    plot_energy,
    DT,
    R_C_B,
    MASS,
    INERTIA,
    I_INV,
    G,
)
from cube_3d_con_tstep import (
    euler_semi_implicit,
    dynamics_falling_ct,
    kin_corners_cs,
)
from transforms_cs import Lq_cs, Aq_cs, H

# solver tolerance
ϵ = 1e-6
# coefficient of friction
MU = 0.1


def smoothnorm(x: cs.SX):
    return cs.sqrt(x.T @ x + ϵ * ϵ) - ϵ


def dynamics_confr_ct(X: cs.SX, F: cs.SX) -> cs.SX:
    """
    Continuous-time SE(3) nonlinear dynamics
    Subject to gravity and 3D collision forces

    :param X: state vector
    :param F: forces (8x3), vector per corner
    """
    Q = X[3:7]  # B to W
    v_w = X[7:10]  # W frame
    ω_b = X[10:13]  # B frame

    # rotation matrix, body to world frame
    A = Aq_cs(Q)

    # get sum of all forces in world frame
    # 3x1 = 3x8 @ 8x1
    ones_nc = cs.SX.ones(8, 1)
    F_w = F.T @ ones_nc  # force in W frame
    tau_b = cs.SX(3, 1)  # torque in B frame
    for i in range(8):
        # add body frame torque due to body frame force
        tau_b += cs.cross(R_C_B[i, :], A.T @ F[i, :].T)

    F_w += np.array([0, 0, -G]) * MASS  # gravity

    dr = v_w
    dq = 0.5 * Lq_cs(Q) @ H @ ω_b
    dv = 1 / MASS * F_w
    # dω = cs.solve(INERTIA, tau_b - cs.cross(ω_b, INERTIA @ ω_b))
    dω = I_INV @ (tau_b - cs.cross(ω_b, INERTIA @ ω_b))
    dX = cs.vertcat(dr, dq, dv, dω)
    return dX


def main():
    n_a = 13  # length of state vector
    n_c = 8  # number of contact points on cube
    # initialize casadi variables
    Xk1 = cs.SX.sym("Xk1", n_a)  # X(k+1), state at next timestep
    F = cs.SX.sym("F", n_c, 3)  # force vector at each corner
    s1 = cs.SX.sym("s1", n_c)  # slack variable 1
    s2 = cs.SX.sym("s2", n_c)  # slack variable 2
    # lagrange mult for magnitude of ground vel per contact point
    lam = cs.SX.sym("lam", n_c)
    X = cs.SX.sym("X", n_a)  # X(k), state

    obj = s1.T @ s1 + s2.T @ s2

    r_c_w_prev = kin_corners_cs(X)  # corner positions at k, 8x3
    r_c_w = kin_corners_cs(Xk1)  # corner positions at k+1, 8x3
    v_tan = ((r_c_w - r_c_w_prev) / DT)[:, 0:2]  # corner xy velocities, 8x2
    z_c_w = r_c_w[:, 2]  # corner heights at k+1, 8x1
    F_tan = F[:, :2]  # tangential ground force friction vectors, 8x2
    F_z = F[:, 2]  # vertical grfs, 8x1

    constr = []  # init constraints

    # --- Equality Constraints --- #
    constr = cs.vertcat(constr, euler_semi_implicit(dynamics_confr_ct, X, F, Xk1) - Xk1)

    # quaternion normalization
    constr = cs.vertcat(constr, cs.norm_2(Xk1[3:7]) ** 2 - 1)

    for i in range(n_c):
        # max dissipation for each corner
        constr = cs.vertcat(
            constr,
            v_tan[i, :].T + lam[i] * F_tan[i, :].T / (smoothnorm(F_tan[i, :].T) + ϵ),
        )

    # --- Inequality Constraints --- #

    # interpenetration
    constr = cs.vertcat(constr, z_c_w)

    for i in range(n_c):
        # primal feasibility friction cone
        constr = cs.vertcat(constr, MU * F_z[i] - smoothnorm(F_tan[i, :].T))

    # interpenetration complementarity
    constr = cs.vertcat(constr, s1 - F_z * z_c_w)

    for i in range(n_c):
        # friction complementarity
        constr = cs.vertcat(
            constr, s2[i] - lam[i] * (MU * F_z[i] - smoothnorm(F_tan[i, :].T))
        )

    opt_variables = cs.vertcat(Xk1, F[:, 0], F[:, 1], F[:, 2], s1, s2, lam)
    lcp = {"x": opt_variables, "p": X, "f": obj, "g": constr}
    opts = {
        "print_time": 0,
        "ipopt.print_level": 0,
        "ipopt.tol": ϵ,
        "ipopt.max_iter": 3000,
    }
    solver = cs.nlpsol("S", "ipopt", lcp, opts)

    n_var = np.shape(opt_variables)[0]
    n_g = np.shape(constr)[0]

    # variable bounds
    ubx = [1e10] * n_var
    lbx = [0] * n_var
    lbx[:n_a] = [-1e10] * n_a  # state can be negative
    lbx[2] = 1  # z pos can't get closer to the ground than 1 m
    lbx[n_a : n_a + n_c * 2] = [-1e10] * (n_c * 2)  # Fx and Fy can be negative

    # constraint bounds
    ubg = [0] * n_g
    ubg[n_a + 1 + n_c * 2 :] = [1e10] * (n_c * 4)  # inequality constraints
    lbg = [0] * n_g

    # initialize simulation variables
    N = 2000  # number of timesteps
    X_0 = np.zeros(n_a)
    X_0[:3] = np.array([0, 0, 3.0])
    X_0[3:7] = np.random.rand(4)
    X_0[3:7] = X_0[3:7] / np.linalg.norm(X_0[3:7])  # normalize the quaternion
    # X_0[3:7] = np.array([1, 0, 0, 0])
    X_0[7:10] = np.array([0, 2, 0])
    X_0[10:13] = np.array([0, -1, 1])

    X_hist = np.zeros((N, n_a))  # state vector for each timestep
    Fx_hist = np.zeros((N, n_c))  # array of corner Fx for each timestep
    Fy_hist = np.zeros((N, n_c))  # array of corner Fy for each timestep
    Fz_hist = np.zeros((N, n_c))  # array of corner Fz for each timestep
    s1_hist = np.zeros((N, n_c))  # array of slack var 1 values for each timestep
    s2_hist = np.zeros((N, n_c))  # array of slack var 2 values for each timestep
    lam_hist = np.zeros((N, n_c))  # array of lambda values for each timestep

    prev_sol = np.hstack((X_0, np.zeros(n_c * 6)))
    X_hist[0, :] = X_0
    for k in tqdm(range(N - 1), desc="Simulating"):
        X_hist[k + 1, :] = rk4_normalized(
            dynamics_falling_ct, X_hist[k, :], np.zeros(3)
        )
        if (kin_corners(X_hist[k + 1, :])[:, 2] <= 0).any() or X_hist[k + 1, 2] <= 1:
            sol = solver(
                x0=prev_sol, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, p=X_hist[k, :]
            )
            X_hist[k + 1, :] = np.reshape(sol["x"][0:n_a], (-1,))
            Fx_hist[k] = np.reshape(sol["x"][n_a : n_a + n_c], (-1,))
            Fy_hist[k] = np.reshape(sol["x"][n_a + n_c : n_a + n_c * 2], (-1,))
            Fz_hist[k] = np.reshape(sol["x"][n_a + n_c * 2 : n_a + n_c * 3], (-1,))
            s1_hist[k] = np.reshape(sol["x"][n_a + n_c * 3 : n_a + n_c * 4], (-1,))
            s2_hist[k] = np.reshape(sol["x"][n_a + n_c * 4 : n_a + n_c * 5], (-1,))
            lam_hist[k] = np.reshape(sol["x"][n_a + n_c * 5 :], (-1,))
            prev_sol = sol["x"]  # sol
        else:
            prev_sol = np.hstack((X_hist[k + 1, :], np.zeros(n_c * 6)))

    name = "cube_3d_confr_tstep"
    hists = {
        "x (m)": X_hist[:, 0],
        "y (m)": X_hist[:, 1],
        "z (m)": X_hist[:, 2],
        "Fz_c (N)": Fz_hist,
        "s1": s1_hist,
        "s2": s2_hist,
        "lam": lam_hist,
    }
    plotting.plot_hist(hists, name)
    animate_cube(X_hist, name)
    plot_energy(X_hist, name)


if __name__ == "__main__":
    main()
