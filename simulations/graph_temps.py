import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.special as sp
from tqdm import tqdm

mpl.rcParams['pcolor.shading'] = 'auto'
font = {'family': 'serif',
        'weight': 'regular',
        'size': 16}
mpl.rc('font', **font)

fig, axs = plt.subplots(
    1, 1,
    #subplot_kw={'projection': 'polar'},
    tight_layout=True,
    figsize=(7.1, 5)
)

def kronecker(i, j):
    return 1 if i == j else 0

pts = 100
R = 1
M = 1; N = 0

radius = np.linspace(0, R, pts)
theta = np.linspace(0, 2 * np.pi, pts)
RR, TT = np.meshgrid(radius, theta)

def amps(m, n, pos, om, q0):
    r0, a0 = pos
    dm0 = kronecker(m, 0)
    kmn = sp.jnp_zeros(m, n + 1)[-1] / R 
    qstar = om * q0
    
    bst = sp.jv(m, kmn * r0) / sp.jv(m, kmn * R)**2
    #print(bst, kmn)
    amn = (2 * qstar * r0 * np.cos(m * a0) * kmn**2) \
            / ((1 + dm0) * np.pi * (k**2 - kmn**2) * ((kmn * R)**2 - m**2)) * bst
    bmn = (2 * qstar * r0 * np.sin(m * a0) * kmn**2) \
            / ((1 - dm0) * np.pi * (k**2 - kmn**2) * ((kmn * R)**2 - m**2)) * bst
    return amn, bmn, kmn

def pressure_field(t, om, q0, pos, phi):
    prt = np.zeros((pts, pts), dtype=complex)
    for m in range(1, M + 1):
        for n in range(0, N + 1):
            amn, bmn, kmn = amps(m, n, pos, om, q0)
            p_given_mode = (amn * np.cos(m * TT) + bmn * np.sin(m * TT)) * sp.jv(m, kmn * RR)
            prt += p_given_mode    
    return prt * np.exp(1j * (om * t - phi))

q0 = 1e-3
c0 = 343
k_cav = (sp.jnp_zeros(M, N + 1)[-1] / R)# + 150
print(k_cav)
k = k_cav + 0.01
om = k * c0

tt = np.linspace(0, 0.03, 300)
pt = []
pt2 = []
for t in tqdm(tt):
    pos1 = (R/2, np.pi/2)
    pos2 = (R/2, 0)
    pos3 = (R/2, 3*np.pi/2)
    pos4 = (R/2, np.pi)


    p1 = pressure_field(t, om, q0, pos1, 0)
    p2 = pressure_field(t, om, q0, pos2, 3 * np.pi / 2)
    p3 = pressure_field(t, om, q0, pos3, np.pi)
    p4 = pressure_field(t, om, q0, pos4, np.pi / 2)
    prt = p1 + p2 + p3 + p4
    pt.append(prt[0, 60].real)
    pt2.append(prt[20, 30].real)


axs.plot(tt, pt / max(pt), label='(1)')
axs.plot(tt, pt2 / max(pt), label='(2)')
axs.legend(loc='lower right')

axs.set_xlabel('Time [s]')
axs.set_ylabel(r'$P/P_0$ [au]')

    # oui = ax.pcolormesh(TT, RR, prt.real, cmap='RdBu_r')
    #fig.colorbar(oui, ax=ax)
    # ax.set_title("t = {:.5f}".format(t))
    # ax.set_theta_direction(-1)
    # ax.set_theta_zero_location("N") # Zero on top (north)
plt.show()