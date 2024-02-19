import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

n_x = 2  # length of state vector
n_u = 1  # length of control vector
m = 10  # mass of the rocket in kg
g = 9.81  # gravitational constant
A = np.array([[0, 1],
              [0, 0]])
B = np.array([[0],
              [1/m]])
G = np.array([[0], 
              [-g]])
dt = 0.001  # timestep size
e = 0.7  # coefficient of restitution

def dynamics_ct(X, U):
    dX = A @ X + B @ U + G.flatten()
    return dX

def lagrangian(z, dz):
    L = 0.5 * m * dz**2 - m * g * z
    return L

def lagrangian_discrete(Xk, X_next):
    Ld = dt * lagrangian((Xk[0] + X_next[0])/2, (Xk[0] + X_next[0])/dt)
    return Ld

def min_problem():
    X = cp.Variable((N+1, n_x))
    U = cp.Variable((N, n_u))    
    for k in range(N-1):
        cost += lagrangian_discrete(X[k], X[k+1])
        # --- calculate constraints --- #
        constr += [X[k] >= 0]
        constr += [grf >= 0]
        constr += [0 == eq3]

    # --- set up solver --- #
    problem = cp.Problem(cp.Minimize(cost), constr)
    problem.solve()  #, verbose=True)
    u = np.zeros(bot.Nu) if u.value is None else u.value
    
N = 5000  # number of timesteps
X_hist = np.zeros((N, n_x))  # array of state vectors for each timestep
X_hist[0, :] = np.array([[1, 0]])
U_hist = np.zeros((N-1, n_u)) # array of control vectors for each timestep



plt.plot(range(N), X_hist[:, 0])
plt.title('Body Position vs Time')
plt.ylabel("z (m)")
plt.xlabel("timesteps")
plt.show()
