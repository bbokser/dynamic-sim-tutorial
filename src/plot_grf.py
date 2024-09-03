import numpy as np
import matplotlib.pyplot as plt

N = 1010
F_hist = np.zeros(N)
z_hist = np.zeros(N)
z = 0.5
for k in range(N):
    z += -0.001
    c = -0.01  # inflection point
    z = np.clip(z, a_min=0, a_max=np.inf)  # clip to just above inflection point
    F_hist[k] = 0.01 / (-c+z)**2  # spring constant inversely related to position
    z_hist[k] = z

plt.plot(z_hist, F_hist)
plt.title('GRF vs Signed Distance')
plt.ylabel("GRF (N)")
plt.xlabel("Signed Distance (m)")
# plt.legend()
# plt.show()
plt.savefig('results/grf_vs_phi.png', dpi=200)
plt.close()