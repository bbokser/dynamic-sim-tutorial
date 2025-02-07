import numpy as np

H = np.zeros((4, 3))
H[1:4, 0:4] = np.eye(3)

T = np.zeros((4, 4))
np.fill_diagonal(T, [1.0, -1.0, -1.0, -1.0])


def hat(w):
    # skew-symmetric
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])


def Lq(Q):
    LQ = np.zeros((4, 4))
    LQ[0, 0] = Q[0]
    LQ[0, 1:4] = -Q[1:4].T
    LQ[1:4, 0] = Q[1:4]
    LQ[1:4, 1:4] = Q[0] * np.eye(3) + hat(Q[1:4])
    return LQ


def Rq(Q):
    RQ = np.zeros((4, 4))
    RQ[0, 0] = Q[0]
    RQ[0, 1:4] = -Q[1:4].T
    RQ[1:4, 0] = Q[1:4]
    RQ[1:4, 1:4] = Q[0] * np.eye(3) - hat(Q[1:4])
    return RQ


def Aq(Q):
    # rotation matrix from quaternion
    return H.T @ Lq(Q) @ Rq(Q).T @ H


def Gq(Q):
    return Lq(Q) @ H


def Ḡq(Q):
    return np.array([np.eye(3), np.zeros((3, 3))], [np.zeros((4, 3)), Gq(Q)])


def cay(ϕ):
    # quaternion from Rodrigues parameters
    # https://roboticexplorationlab.org/papers/planning_with_attitude.pdf
    return (1 / np.sqrt(1 + ϕ.T @ ϕ)) @ np.array([1, ϕ]).T


def cay_inv(Q):
    # Rodrigues parameters from quaternion
    qw = Q[0]
    qv = Q[1:]
    return qw / qv


def quat_to_axis_angle(Q):
    qw = Q[0]
    qx = Q[1]
    qy = Q[2]
    qz = Q[3]
    if qw == 1:
        return np.zeros(3), 2 * np.acos(qw)
    else:
        angle = 2 * np.acos(qw)
        x = qx / np.sqrt(1 - qw * qw)
        y = qy / np.sqrt(1 - qw * qw)
        z = qz / np.sqrt(1 - qw * qw)
        return np.array([x, y, z]), angle
