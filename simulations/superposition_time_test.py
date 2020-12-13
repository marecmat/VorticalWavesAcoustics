import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.special as sp
from tqdm import tqdm
mpl.rcParams['pcolor.shading'] = 'auto'

m = 1; n = 1                                    # mode to study in the cavity
pts = 1000                                      # Spatial resolution
R = 0.35                                        # Radius of the cavity
c0 = 340                                        # speed of sound in the cavity
r0 = 0.25                                       # radial position of the point source in the cavity 
q0 = 1e-2                                       # driving force of the point source
t0 = [0, 2 * np.pi / 3, 4 * np.pi / 3]          # angle to locate the cavity (neglected here)

f = np.linspace(0.1, 2000, pts)
om = 2 * np.pi * f
k = om / c0

# Zeros of the derivative of Bessel functions == modes of the system
kmn = sp.jnp_zeros(m, n + 1)[-1] / R # 1.84/R #
theta = np.arange(0, 2 * np.pi, 0.01 * np.pi)
r = np.linspace(0, R, pts)

fig, axs = plt.subplots(
    2, 4,
    subplot_kw={'projection': 'polar'},
    tight_layout=True,
    figsize=(5, 5)
)
superp_fields = np.zeros((r.size, theta.size), dtype=complex)
allfields = []
for ax, timev in tqdm(zip(axs.flat, [0, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1])):
    for theta0 in t0:
        fc = np.zeros((r.size, theta.size), dtype=complex)
        ax.plot(theta0, r0, 'bo', label='source')

        amp = (2 * 1j * om * q0 * np.sin(m * theta0) * r0 / k**2)
        for t in range(len(theta)):
            bess_ratio = sp.jv(m, kmn * r0) / sp.jv(m, kmn * R)**2
            sumterm = np.array([(2 / (kmn**2 - k**2)) * bess_ratio * (1 / (1 - (m**2 / (kmn**2 * R**2)))) for m in range(m + 1)], dtype=complex).sum()
            amplterm = amp * (1 / k**2) + sumterm
            fgt = amplterm * sp.jv(m, kmn * r) * np.cos(m * theta[t])
            fc[:, t] = fgt.real * np.exp(1j * om * timev)
        
        allfields.append(fc)
        superp_fields = np.add(superp_fields, fc)

    print(superp_fields.shape, fc.shape)
    cmesh = ax.pcolormesh(theta, r, np.real(superp_fields), cmap='RdBu_r')
    fig.colorbar(cmesh, ax=ax)
    ax.set_title("time {}".format(timev))
    ax.set_rlabel_position(np.pi/2)
    # ax.set_rlim(0.04, 0.06)
    # ax.set_rlim(0.14, 0.152)
    # ax.set_rlim(R-0.01, R)
    ax.set_theta_zero_location("N") # Zero on top (north)
    ax.grid(True)


#fig, ax = plt.subplots(1, 1)
#ax.semilogy(r0, 1, 'bo', label='source')
#ax.semilogy(r, fc[:, 100], label=r'$\pi$')
#ax.semilogy(r, fc[:, 0], label=r'0')
#ax.semilogy(r, fc[:, 50], label=r'$\pi/2$')
#ax.semilogy(r, fc[:, 25], label=r'$\pi/4$')
#ax.semilogy(r, fc[:, 150], label=r'$3\pi/2$')
#ax.legend()

#fig, ax = plt.subplots(1, 3,
#    subplot_kw={'projection': 'polar'},
#    tight_layout=True,
#    figsize=(5, 5)
#)
#
#for i in range(len(allfields)):
#    ax[i].set_title(r'$\theta_0$ {:.2f}'.format(t0[i]))
#    ax[i].pcolormesh(theta, r, np.real(allfields[i]), cmap='jet')

plt.show()
