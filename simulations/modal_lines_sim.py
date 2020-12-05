import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.special as sp
from tqdm import tqdm
mpl.rcParams['pcolor.shading'] = 'auto'

m = 2; n = 0     # mode to study in the cavity
pts = 1000      # Spatial resolution
R = 0.35        # Radius of the cavity
c0 = 340        # speed of sound in the cavity
r0 = 0.192       # radial position of the point source in the cavity 
t0 = 0          # angle to locate the cavity (neglected here)
q0 = 1e-4          # driving force of the point source

f = np.linspace(0.1, 2000, pts)
om = 2 * np.pi * f
k = om / c0

# Zeros of the derivative of Bessel functions == modes of the system
kmn = sp.jnp_zeros(m, n + 1)[-1] / R # 1.84/R #
theta = np.arange(0, 2 * np.pi, 0.01 * np.pi)
r = np.linspace(0.1, R, pts)

fig, ax = plt.subplots(
    subplot_kw={'projection': 'polar'},
    tight_layout=True,
    figsize=(5, 5)
)

fc = np.zeros((r.size, theta.size))
amp = (2 * 1j * om * q0 * r0 / k**2)
for t in tqdm(range(len(theta))):
    bess_ratio = sp.jv(m, kmn * r0) / sp.jv(m, kmn * R)**2
    sumterm = (2 / (kmn**2 - k**2)) * bess_ratio * (1 / (1 - (m**2 / (kmn**2 * R**2))))
    amplterm = amp * (-1 / k**2) + sumterm
    fc[:, t] = amplterm * sp.jv(m, kmn * r) * np.cos(m * theta[t])

cmesh = ax.pcolormesh(theta, r, np.real(fc), cmap='jet')
fig.colorbar(cmesh)
ax.plot(t0, r0, 'bo', label='source')
ax.set_title("k{}{}R = {:.2f}".format(m, n, kmn * R))
ax.set_rlabel_position(np.pi/2)
# ax.set_rlim(0.04, 0.06)
# ax.set_rlim(0.14, 0.152)
# ax.set_rlim(R-0.01, R)
ax.set_theta_zero_location("N") # Zero on top (north)
ax.grid(True)


fig, ax = plt.subplots(1, 1)
ax.semilogy(r0, 1, 'bo', label='source')
ax.semilogy(r, fc[:, 100], label=r'$\pi$')
ax.semilogy(r, fc[:, 0], label=r'0')
ax.semilogy(r, fc[:, 50], label=r'$\pi/2$')
ax.semilogy(r, fc[:, 25], label=r'$\pi/4$')
ax.semilogy(r, fc[:, 150], label=r'$3\pi/2$')

ax.legend()
plt.show()