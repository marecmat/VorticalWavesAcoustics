import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.special as spl
from tqdm import tqdm
mpl.rcParams['pcolor.shading'] = 'auto'

# polar plots doc
# https://matplotlib.org/api/projections_api.html

# Test to find back results from the last slide of lecture 3
pts = 1000
a = 3
m = 1
kwmn = 1 / a

theta = np.linspace(0, 2 * np.pi, pts)
radius = np.linspace(0, a, pts)
fig, ax = plt.subplots(
    subplot_kw={'projection': 'polar'},
    tight_layout=True,
    figsize=(5, 5)
)

fc = np.zeros((radius.size, theta.size))
for t in tqdm(range(len(theta))):
    fc[:, t] = spl.jv(m, kwmn * radius) * np.cos(m * theta[t])

cmesh = ax.pcolormesh(theta, radius, fc, cmap='jet')
fig.colorbar(cmesh)
ax.set_rlabel_position(np.pi/2)
ax.set_theta_zero_location("N") # Zero on top (north)
ax.grid(True)

plt.show()