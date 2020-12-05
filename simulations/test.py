import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.special as spl
from tqdm import tqdm
mpl.rcParams['pcolor.shading'] = 'auto'

# polar plots doc
# https://matplotlib.org/api/projections_api.html

# Test to find back results from the last slide of lecture 3
pts = 1000 # Spatial resolution
r0 = 1 # Position of the point source in the cavity 
theta0 = 0 #
q0 = 1e-4
R = 2 # Radius of the cavity
# a = 3
m = 1

# Zeros of the derivative of Bessel functions
kmn = 1.84/R #spl.jnyn_zeros(m, m + 1)[1][-1]
c0 = 340
f = np.linspace(0, 2000, pts)
om = 2 * np.pi * f
k = om / c0
theta = np.linspace(0, 2 * np.pi, pts)
r = np.linspace(0, R, pts)

fig, ax = plt.subplots(
    subplot_kw={'projection': 'polar'},
    tight_layout=True,
    figsize=(5, 5)
)

fc = np.zeros((r.size, theta.size))
for t in tqdm(range(len(theta))):
    fc[:, t] = spl.jv(m, kmn * r) * np.cos(m * theta[t])
cmesh = ax.pcolormesh(theta, r, np.real(fc), cmap='jet')
fig.colorbar(cmesh)
ax.set_rlabel_position(np.pi/2)
# ax.set_rlim(0.19, 0.22)
ax.set_theta_zero_location("N") # Zero on top (north)
ax.grid(True)

plt.show()