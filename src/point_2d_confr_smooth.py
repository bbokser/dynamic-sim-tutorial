import numpy as np
import matplotlib.pyplot as plt

import os_utils

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
mu = 0.1  # coefficient of friction

def get_forces(X):
    z = X[1]
    dx = X[2]
    dz = X[3]
    c = -0.01  # inflection point
    phi = np.clip(z, a_min=c+0.005, a_max=np.inf)  # signed distance. clip to just above inflection point
    distance_fn = 1 / (-c+phi)**2  # y = 1/x^2 relation
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

N = 5000  # number of timesteps
X_hist = np.zeros((N, n_x))  # array of state vectors for each timestep
Fx_hist = np.zeros(N)  # array of x GRF forces for each timestep
Fz_hist = np.zeros(N)  # array of z GRF forces for each timestep
X_hist[0, :] = np.array([[0, 1, 1, 0]])
U_hist = np.zeros((N-1, n_u)) # array of control vectors for each timestep

for k in range(N-1):
    Fx_hist[k], Fz_hist[k] = get_forces(X_hist[k, :])  # get spring-damper force
    U_hist[k, 0] += Fx_hist[k]  # add friction force to x component of control vector
    U_hist[k, 1] += Fz_hist[k]  # add grf to z component of control vector
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
file_name = '2d_confr_smooth'
path_dir_imgs, path_dir_gif = os_utils.prep_animation()
frames = 40  # save a snapshot every X frames
j = 0
for k in range(N-1)[::frames]:
    plt.xlim([-0, 2])
    plt.ylim([-0, 2])
    plt.title('Position vs Time')
    plt.xlabel('x (m)')
    plt.ylabel('z (m)')
    plt.scatter(X_hist[k, 0], X_hist[k, 1])
    plt.text(0.01, 0, 't = ' + '{:.2f}'.format(round(k * dt, 2)) + 's', ha='left', va='bottom', transform=plt.gca().transAxes)
    plt.savefig(path_dir_imgs + '/' + str(j).zfill(4) + '.png')
    plt.close()
    j += 1

os_utils.convert_gif(path_dir_imgs=path_dir_imgs, path_dir_output=path_dir_gif, file_name=file_name)