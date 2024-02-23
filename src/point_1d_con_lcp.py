import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
import cvxpy as cp

n_x = 2  # length of state vector
n_u = 2  # length of control vector
m = 10  # mass of the rocket in kg
g = 9.81  # gravitational constant
A = np.array([[0, 1],
              [0, 0]])
B = np.array([[0, 0],
              [1/m, 1/m]])
G = np.array([[0], 
              [-g]])
dt = 0.001  # timestep size
e = 0.7  # coefficient of restitution

# Discretize by matrix exponential method
ABG = np.vstack((np.hstack((A, B, G)), np.zeros((n_u + 1, n_x + n_u + 1))))
M = expm(ABG * dt)
Ad = M[0:n_x, 0:n_x]
Bd = M[0:n_x, n_x:n_x + n_u]
Gd = M[0:n_x, n_x + n_u:]  

N = 5000  # number of timesteps
X = cp.Variable((N, n_x))
U = cp.Variable((N-1, n_u))   
u = U[:, 0]
lam = U[:, 1]
z = X[:, 0]
cost = 0  # init cost
constr = []  # init constraints
for k in range(N-1):
    cost += lam[k]
    # --- calculate constraints --- #
    constr += [X[k+1, :] == Ad @ X[k, :] + Bd @ U[k, :] + Gd.flatten(),  # dynamics
               z[k] >= 0,  # position can not be negative (underground)
               lam[k] >= 0,  # GRF cannot be negative (pulling)
            #    lam[k] * z[k] == 0,  # GRF = 0 when not touching the ground
               u[k] == 0]  # no control for now

constr += [X[0, :] == np.array([1, 0])] # initial conditions
# --- set up solver --- #
problem = cp.Problem(cp.Minimize(cost), constr)
problem.solve(verbose=True)

grf = U.value[:, 1]
X_hist = X.value  # array of state vectors for each timestep

plt.plot(range(N), X_hist[:, 0])
plt.title('Body Position vs Time')
plt.ylabel("z (m)")
plt.xlabel("timesteps")
plt.show()
