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

fig, axs = plt.subplots(
    1, 1,
    subplot_kw={'projection': 'polar'},
    tight_layout=False,
    figsize=(7.1, 5)
)

def kronecker(i, j):
    return 1 if i == j else 0

pts = 100
R = 1
M = 1; N = 0 # modes to observe in the cavity
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
k = k_cav + 0.01
om = k * c0

ax = axs
for t, cm in zip([1.3e-3, 1.8e-3, 2.3e-3, 2.8e-3], ['Purples', 'Blues', 'Greens', 'Oranges']):
    pos1 = (R/2, np.pi/2)
    pos2 = (R/2, 0)
    pos3 = (R/2, 3*np.pi/2)
    pos4 = (R/2, np.pi)
    p1 = pressure_field(t, om, q0, pos1, 0)
    ax.plot(pos1[1], pos1[0], 'ko', mfc='none')#, label='source 1')
    p2 = pressure_field(t, om, q0, pos2, 3 * np.pi / 2)
    ax.plot(pos2[1], pos2[0], 'ko', mfc='none')#, label='source 2')
    p3 = pressure_field(t, om, q0, pos3, np.pi)
    ax.plot(pos3[1], pos3[0], 'ko', mfc='none')#, label='source 1')
    p4 = pressure_field(t, om, q0, pos4, np.pi / 2)
    ax.plot(pos4[1], pos4[0], 'ko', mfc='none')#, label='source 2')

    ax.plot(0, 0.6, 'k+', markersize=8)
    ax.plot(2 * np.pi / 5, 0.3, 'k+', markersize=8)
    # pos3 = (R/2, 4 * np.pi / 3)
    # p3 = pressure_field(t, om, q0, pos3, pos2[1])
    # ax.plot(pos3[1], pos3[0], 'mo', label='source 3')
    prt = p1 + p2 + p3 + p4
    prt = prt.real / np.max(prt.real)
    # ax.plot(theta[0], radius[50], 'm*')

    # ax.plot(0, 0, 'k', label='t = {} s'.format(t)) # for legend
    oui = ax.contour(TT, RR, prt, 4, cmap=cm)
    # fig.colorbar(oui, ax=ax)
    # ax.clabel(oui, fontsize=11, inline=1, fmt='%1.1f')
    # ax.set_title("t = {:.4f} s".format(t))
    ax.set_rticks([])
    # ax.set_rlabel_position(-np.pi/2)
    ax.set_xticks(np.pi/180. * np.arange(0, 359, 90))
    ax.set_theta_direction(-1)
    ax.set_theta_zero_location("N") # Zero on top (north)
    # ax.legend()
    #ax.grid(True)

plt.show()