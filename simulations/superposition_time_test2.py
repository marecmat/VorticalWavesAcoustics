import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.special as sp
from tqdm import tqdm
import matplotlib.animation as animation
mpl.rcParams['pcolor.shading'] = 'auto'

m = 1; n = 1                                    # mode to study in the cavity
pts = 1000                                      # Spatial resolution
R = 0.35                                        # Radius of the cavity
c0 = 340                                        # speed of sound in the cavity
r0 = 0.25                                       # radial position of the point source in the cavity 
q0 = 1e-2                                       # driving force of the point source
t0 = [0, 2 * np.pi / 3, 5 * np.pi / 3]          # angle to locate the cavity (neglected here)
phis = [0, 2 * np.pi / 3, 5 * np.pi / 3]          # angle to locate the cavity (neglected here)

f = np.linspace(0.1, 2000, pts)
om = 2 * np.pi * f
k = om / c0

# Zeros of the derivative of Bessel functions == modes of the system
kmn = sp.jnp_zeros(m, n + 1)[-1] / R # 1.84/R #
theta = np.arange(0, 2 * np.pi, 0.01 * np.pi)
r = np.linspace(0, R, pts)
print(len(theta), len(r))

fig, ax = plt.subplots(
    1,1,
    # subplot_kw={'projection': 'polar'},
    tight_layout=False,
    figsize=(5, 5)
)

def pressf(timev):
    superp_fields = np.zeros((r.size, theta.size), dtype=complex)
    allfields = []
    for theta0, phi0 in zip(t0, phis):
        fc = np.zeros((r.size, theta.size), dtype=complex)
        ax.plot(theta0, r0, 'bo', label='source')

        amp = (2 * 1j * om * q0 * np.sin(m * theta0) * r0 / k**2)
        for t in range(len(theta)):
            bess_ratio = sp.jv(m, kmn * r0) / sp.jv(m, kmn * R)**2
            sumterm = np.array([(2 / (kmn**2 - k**2)) * bess_ratio * (1 / (1 - (m**2 / (kmn**2 * R**2)))) for m in range(m + 1)], dtype=complex).sum()
            amplterm = amp * (1 / k**2) + sumterm
            fgt = amplterm * sp.jv(m, kmn * r) * np.cos(m * theta[t])
            fc[:, t] = fgt.real * np.exp(1j * om * timev + phi0)
        
        allfields.append(fc)
        superp_fields = np.add(superp_fields, fc)
    return superp_fields

timev = 0
cmesh = ax.imshow(np.real(pressf(timev)), animated=True)#, cmap='RdBu_r')

def updatefig(*args):
    global timev
    timev += 0.0001
    cmesh.set_array(np.real(pressf(timev)))
    return cmesh,

fig.colorbar(cmesh, ax=ax)
#ax.set_title("time {}".format(timev))
#ax.set_rlabel_position(np.pi/2)
#ax.set_theta_zero_location("N") # Zero on top (north)
#ax.grid(True)

ani = animation.FuncAnimation(fig, updatefig, interval=50, blit=True)
plt.show()