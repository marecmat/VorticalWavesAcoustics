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
    2, 2,
    subplot_kw={'projection': 'polar'},
    tight_layout=True,
    figsize=(7.1, 5)
)

def kronecker(i, j):
    return 1 if i == j else 0

def nearid(array, value):
    return (np.abs(np.asarray(array) - value)).argmin()


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
    qstar = -1j * om * q0
    
    bst = sp.jv(m, kmn * r0) / sp.jv(m, kmn * R)**2
    #print(bst, kmn)
    amn = (2 * qstar * r0 * np.cos(m * a0) * kmn**2) \
            / ((1 + dm0) * np.pi * (k**2 - kmn**2) * ((kmn * R)**2 - m**2)) * bst
    bmn = (2 * qstar * r0 * np.sin(m * a0) * kmn**2) \
            / ((1 - dm0) * np.pi * (k**2 - kmn**2) * ((kmn * R)**2 - m**2)) * bst
    return amn, bmn, kmn

def pressure_field(M, N, om, q0, pos, phi):
    t = 1e-5
    prt = np.zeros((pts, pts), dtype=complex)
    for m in range(1, M + 1):
        for n in range(0, N + 1):
            amn, bmn, kmn = amps(m, n, pos, om, q0)
            p_given_mode = (amn * np.cos(m * TT) + bmn * np.sin(m * TT)) * sp.jv(m, kmn * RR)
            prt += p_given_mode    
    return prt * np.exp(1j * (om * t - phi))

q0 = 1e-2
c0 = 343
M = 2; N = 1

letters = ['a', 'b', 'c', 'd']
pos = [
    (0.50, np.pi),
    (0.75, np.pi/4),
    (0.20, 5*np.pi/4),
    (0.1, np.pi/2)
]

for let, pos1, ax in zip(letters, pos, axs.flat):
    k_cav = (sp.jnp_zeros(M, N + 1)[-1] / R)# + 150
    k = k_cav + 0.05 * k_cav
    om = k * c0

    p = pressure_field(M, N, om, q0, pos1, 0)
    ps = p[nearid(radius, pos1[0]), nearid(theta, pos1[1])]
    p = p / ps #/ p[nearid(radius, pos1[0]), nearid(theta, pos1[1])]

    ax.plot(pos1[1], pos1[0], 'ko', mfc='none')
    prt = p # + p2 + p3 + p4

    oui = ax.pcolormesh(TT, RR, prt.real, cmap='RdBu_r')
    fig.colorbar(oui, ax=ax)
    ax.set_title(r"{}.".format(let))
    print(let, pos1[0], pos1[1])
    ax.set_rticks([R/2, R])
    ax.set_rlabel_position(270)

    # ax.set_rlabel_position(-np.pi/2)
    # ax.set_xticks(np.pi/180. * np.arange(45, 316, 90))
    ax.set_xticks([np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4])
    ax.set_xticklabels([r"$\pi$/4", r"3$\pi$/4", r"5$\pi$/4", r"7$\pi$/4"])
    ax.set_theta_direction(-1)
    ax.set_theta_zero_location("N") # Zero on top (north)
    ax.grid(True)

plt.show()