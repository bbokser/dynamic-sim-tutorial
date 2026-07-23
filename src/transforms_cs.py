import casadi as cs
from transforms import hat, H
import numpy as np


def Lq_cs(Q: cs.SX) -> cs.SX:
    LQ = cs.SX(4, 4)
    LQ[0, 0] = Q[0]
    LQ[0, 1:4] = -Q[1:4].T
    LQ[1:4, 0] = Q[1:4]
    LQ[1:4, 1:4] = Q[0] * np.eye(3) + hat(Q[1:4])
    return LQ


def Rq_cs(Q: cs.SX) -> cs.SX:
    RQ = cs.SX(4, 4)
    RQ[0, 0] = Q[0]
    RQ[0, 1:4] = -Q[1:4].T
    RQ[1:4, 0] = Q[1:4]
    RQ[1:4, 1:4] = Q[0] * np.eye(3) - hat(Q[1:4])
    return RQ


def Aq_cs(Q):
    # rotation matrix from quaternion
    return H.T @ Lq_cs(Q) @ Rq_cs(Q).T @ H
