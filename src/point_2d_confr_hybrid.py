import numpy as np
import matplotlib.pyplot as plt

n_x = 4  # length of state vector
n_u = 2  # length of control vector
m = 10  # mass of the rocket in kg
A = np.array([[0, 0, 1, 0],
              [0, 0, 0, 1],
              [0, 0, 0, 0],
              [0, 0, 0, 0]])
B = np.array([[0,   0],
              [0,   0],
              [1/m, 0],
              [0, 1/m]])
G = np.array([[0, 0, 0, -9.81]]).T
dt = 0.001  # timestep size
e = 0.7  # coefficient of restitution
mu = 0.5  # coefficient of friction

def dynamics_ct(X, U):
    dX = A @ X + B @ U + G.flatten()
    return dX

def integrator_euler(dyn_ct, xk, uk):
    X_next = xk + dt * dyn_ct(xk, uk)
    return X_next

def jump_map(X):
    X[1] = 0  # reset z position to zero
    dx_before = X[2]  # x velocity before impact
    dz_before = X[3]  # z velocity before impact
    dz_after = -e * dz_before  # reverse velocity and multiply by coefficient of restitution
    a_z = (dz_after - dz_before)/dt  # acceleration
    Fz = m * a_z  # get ground reaction force
    Fx = mu * Fz * np.sign(dx_before)  # calculate max friction force from GRF normal
    a_x = Fx / m
    # don't let friction change the object's direction--that's not possible
    a_x = np.clip(a_x * dt, -dx_before, 0) / dt
    Fx = a_x * m  # update Fx
    dx_after = a_x * dt + dx_before
    X[2] = dx_after  # x velocity after impact
    X[3] = dz_after  # z velocity after impact
    return X, Fx, Fz

N = 5000  # number of timesteps
X_hist = np.zeros((N, n_x))  # array of state vectors for each timestep
Fx_hist = np.zeros((N, 1))  # array of x GRF forces for each timestep
Fz_hist = np.zeros((N, 1))  # array of z GRF forces for each timestep
X_hist[0, :] = np.array([[0, 1, 1, 0]])
U_hist = np.zeros((N-1, n_u)) # array of control vectors for each timestep

for k in range(N-1):
    if X_hist[k, 1] < 0:  # guard function
        X_hist[k, :], Fx_hist[k, :], Fz_hist[k, :] = jump_map(X_hist[k, :])  # dynamics rewrite based on impact
    X_hist[k+1, :] = integrator_euler(dynamics_ct, X_hist[k, :], U_hist[k, :])

# plot w.r.t. time
fig, axs = plt.subplots(2, sharex='all')
fig.suptitle('Body Position vs Time')
plt.xlabel('timesteps')
axs[0].plot(range(N), X_hist[:, 1])
axs[0].set_ylabel('z (m)')
axs[1].plot(range(N), Fz_hist)
axs[1].set_ylabel('z GRF (N)')
plt.show()

# plot in cartesian coordinatesS
plt.plot(X_hist[:, 0], X_hist[:, 1])
plt.title('Body Position in the XZ plane')
plt.xlabel("x (m)")
plt.ylabel("z (m)")
plt.show()

# generate animation
frames = 40  # save a snapshot every X frames
j = 0
for k in range(N-1)[::frames]:
    plt.xlim([-1, 1])
    plt.ylim([-0, 2])
    plt.title('Position vs Time')
    plt.xlabel('x (m)')
    plt.ylabel('z (m)')
    plt.scatter(X_hist[k, 0], X_hist[k, 1])
    plt.text(0.01, 0, 't = ' + '{:.2f}'.format(round(k * dt, 2)) + 's', ha='left', va='bottom', transform=plt.gca().transAxes)
    plt.savefig('imgs/' + str(j).zfill(4) + '.png')
    plt.close()
    j += 1
