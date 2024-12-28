import numpy as np
import matplotlib.pyplot as plt
from point_1d_con_smooth import get_grf

N = 1000
F_hist = np.zeros(N)
z_hist = np.zeros(N)
X = np.zeros(2)
k = 0
for z in np.linspace(0.5, -0.05, N):
    X[0] = z
    F_hist[k] = get_grf(X)
    z_hist[k] = X[0]
    k += 1

plt.plot(z_hist, F_hist)
plt.title("GRF vs Signed Distance")
plt.ylabel("GRF (N)")
plt.xlabel("Signed Distance (m)")
# plt.legend()
# plt.show()
fig = plt.gcf()
fig.tight_layout()
fig.set_size_inches(10, 10)
plt.savefig("results/grf_vs_phi.png", dpi=200)
plt.close()
