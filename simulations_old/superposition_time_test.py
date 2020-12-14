import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.special as sp
from tqdm import tqdm
mpl.rcParams['pcolor.shading'] = 'auto'

m = 1; n = 0                                    # mode to study in the cavity
pts = 100                                      # Spatial resolution
R = 1                                        # Radius of the cavity
c0 = 340                                        # speed of sound in the cavity
r0 = 0.5                                       # radial position of the point source in the cavity 
q0 = 1000                                       # driving force of the point source
t0 = [0, 2 * np.pi / 3, 4 * np.pi / 3]          # angle to locate the cavity (neglected here)
phis = [0, 2 * np.pi / 3, 4 * np.pi / 3]
colors = ['b', 'g', 'r']

# Zeros of the derivative of Bessel functions == modes of the system
kmn = sp.jnp_zeros(m, n + 1)[-1] / R # 1.84/R #
om = kmn * c0
k = kmn
theta = np.arange(0, 2 * np.pi, pts)
r = np.linspace(0, R, pts)
#k = kmn
#om = k * c0


fig, axs = plt.subplots(
    2, 5,
    subplot_kw={'projection': 'polar'},
    tight_layout=False,
    figsize=(5, 5)
)
superp_fields = np.zeros((r.size, theta.size), dtype=complex)
allfields = []
for ax, timev in tqdm(zip(axs.flat, [0.0001, 0.0005, 0.001, 0.002, 0.003, 0.004, 0.005, 0.01, 0.05, 0.1])):
    for theta0, phi0, c in zip(t0, phis, colors):
        fc = np.zeros((r.size, theta.size), dtype=complex)
        ax.plot(theta0, r0, '{}o'.format(c), mfc='none', label='source')

        for t in range(len(theta)):
            amp = (2 * 1j * r0 * om * q0 * np.sin(m * (t - theta0)) / k**2)
            bess_ratio = sp.jv(m, kmn * r0) / sp.jv(m, kmn * R)**2
            sumterm = np.array([(2 / (kmn**2 - k**2)) * amp * bess_ratio * (1 / (1 - (m**2 / (kmn**2 * R**2)))) for m in range(m + 1)], dtype=complex).sum()
            amplterm = (1 / k**2) + sumterm
            fgt = amplterm * sp.jv(m, kmn * r) * np.cos(m * theta[t])
            fc[:, t] = fgt.real * np.exp(1j * om * timev + phi0)

        # amp = (2 * 1j * om * q0 * np.sin(m * theta0) * r0 / k**2)
        # for t in range(len(theta)):
        #     bess_ratio = sp.jv(m, kmn * r0) / sp.jv(m, kmn * R)**2
        #     sumterm = np.array([(2 / (kmn**2 - k**2)) * bess_ratio * (1 / (1 - (m**2 / (kmn**2 * R**2)))) for m in range(m + 1)], dtype=complex).sum()
        #     amplterm = amp * (1 / k**2) + sumterm
        #     fgt = amplterm * sp.jv(m, kmn * r) * np.cos(m * theta[t])
        #     fc[:, t] = fgt.real * np.exp(1j * (om * timev) - phi0)
        
        allfields.append(fc)
        superp_fields = np.add(superp_fields, fc)

    cmesh = ax.pcolormesh(theta, r, np.real(superp_fields), cmap='RdBu_r')
    #fig.colorbar(cmesh, ax=ax)
    ax.set_title("time {}".format(timev))
    ax.set_rlabel_position(np.pi/2)
    ax.set_theta_direction(-1)
    ax.set_theta_zero_location("N") # Zero on top (north)
    ax.grid(True)

plt.show()
