import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.special as sp
from tqdm import tqdm
mpl.rcParams['pcolor.shading'] = 'auto'
font = {'family': 'serif',
        'weight': 'regular',
        'size': 12}
mpl.rc('font', **font)

m = 1; n = 1     # mode to study in the cavity
pts = 100      # Spatial resolution
R = 0.35        # Radius of the cavity
c0 = 340        # speed of sound in the cavity
#r0 = 0.192       # radial position of the point source in the cavity 
t0 = 0          # angle to locate the cavity (neglected here)
q0 = 1e-6          # driving force of the point source
theta = np.arange(0, 2 * np.pi, 0.01 * np.pi)
r = np.linspace(0, R, pts)
sources = [   # even number of sources pls
    #('a)', 0.25, 0), 
    #('b)', 0.13, np.pi/4),
    ('a)', 0.12, np.pi),
    ('b)', 0.25, 3 * np.pi / 2),
    #('e)', 0.3, 0), 
    #('f)', 0.35, 3 * np.pi / 4),
    ('c)', 0.3, np.pi), 
    ('d)', 0.28, np.pi / 2)
]

f = np.linspace(0.1, 2000, pts)
om = 2 * np.pi * f
k = om / c0

def find_nearest(array, value):
    return (np.abs(np.asarray(array) - value)).argmin()

fig, axs = plt.subplots(
    2, 2,
    subplot_kw={'projection': 'polar'},
    tight_layout=True,
    figsize=(7.1, 5)
)

for ax, pos in zip(axs.flat, sources):
    label, r0, t0 = pos
    # Zeros of the derivative of Bessel functions == modes of the system
    kmn = sp.jnp_zeros(m, n + 1)[-1] / R # 1.84/R #
    fc = np.zeros((r.size, theta.size))
    amp = (2 * 1j * om * q0 * r0 * np.sin(m * t0) / k**2)
    for t in range(len(theta)):
        bess_ratio = sp.jv(m, kmn * r0) / sp.jv(m, kmn * R)**2
        # sumterm = (2 / (kmn**2 - k**2)) * bess_ratio * (1 / (1 - (m**2 / (kmn**2 * R**2))))
        sumterm = np.array([(2 / (kmn**2 - k**2)) * bess_ratio * (1 / (1 - (m**2 / (kmn**2 * R**2)))) for m in range(m + 1)]).sum()
        amplterm = amp * (1 / k**2) + sumterm
        fc[:, t] = amplterm * sp.jv(m, kmn * r) * np.cos(m * theta[t])
    
    # pressure field at the source:
    psource = fc[
        find_nearest(r, r0), 
        find_nearest(theta, t0)
    ].real
    cmesh = ax.pcolormesh(theta, r, fc.real / 3, cmap='RdBu_r') # / float(psource)
    fig.colorbar(cmesh, ax=ax)
    ax.plot(t0, r0, 'ko', mfc='none', label='source')
    ax.set_title(r"{} ({},{})".format(label, r0, int(t0 * (180/np.pi))))
    ax.set_rlabel_position(270)
    ax.set_rticks([0, round(max(r)/2, 2), max(r)])
    ax.set_theta_direction(-1)
    ax.set_xticks(np.pi/180. * np.arange(45, 316, 90))
    ax.set_theta_zero_location("N") # Zero on top (north)
    ax.grid(True)

plt.show()
