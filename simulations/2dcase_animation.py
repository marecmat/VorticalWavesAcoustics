import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation
import scipy.special as sp
from tqdm import tqdm

mpl.rcParams['pcolor.shading'] = 'auto'
font = {'family': 'serif',
        'weight': 'regular',
        'size': 12}
mpl.rc('font', **font)

fig, ax = plt.subplots(
    1, 1,
    subplot_kw={'projection': 'polar'},
    tight_layout=False,
    figsize=(7.1, 5)
)

def kronecker(i, j):
    return 1 if i == j else 0

def amps(m, n, pos, om, q0):
    r0, a0 = pos
    dm0 = kronecker(m, 0)
    kmn = sp.jnp_zeros(m, n + 1)[-1] / R 
    k = om / c0
    qstar = om * q0
    
    bst = sp.jv(m, kmn * r0) / sp.jv(m, kmn * R)**2
    #print(bst, kmn)
    amn = (2 * qstar * np.cos(m * a0) * kmn**2) \
            / ((1 + dm0) * np.pi * (k**2 - kmn**2) * ((kmn * R)**2 - m**2)) * bst
    bmn = (2 * qstar * np.sin(m * a0) * kmn**2) \
            / ((1 - dm0) * np.pi * (k**2 - kmn**2) * ((kmn * R)**2 - m**2)) * bst
    return amn, bmn, kmn

def pressure_field(t, om, q0, pos, phi):
    prt = np.zeros((pts, pts), dtype=complex)
    for m in range(1, M + 1):
        for n in range(0, N + 1):
    # m = M; n = N
            amn, bmn, kmn = amps(m, n, pos, om, q0)
            p_given_mode = (amn * np.cos(m * TT) + bmn * np.sin(m * TT)) * sp.jv(m, kmn * RR)
            prt += p_given_mode    
    return prt * np.exp(1j * (om * t - phi))

# cavity parameters
R = 1                                       # Radius of the cavity
c0 = 343                                    # sound velocity (m/s)
M = 6; N = 2                                # modes to observe in the cavity
print("mode: ", M, N)
pts    = 100                                # number of points contained in the matrix 
k_cav  = sp.jnp_zeros(M, N + 1)[-1] / R
radius = np.linspace(0, R, pts)
theta  = np.linspace(0, 2 * np.pi, pts)
RR, TT = np.meshgrid(radius, theta)

# source parameters
q0 = 1e-5                                   # mass flow rate of the source
om = (k_cav + 0.0001) * c0                  # excitation frequency of the source
rr = 3*R/4                                  # radial position
nb_source = 3

pr_time = []
time = np.arange(0, 10e-3, 1e-4)

poss = [(rr, 0), (rr, np.pi/2)]
phases = [0, 2*np.pi/3]

for t in tqdm(time):
    prt = np.zeros((pts, pts), dtype=complex)
    for n in range(nb_source):
        pos = (rr, 2 * n * np.pi / nb_source)
        phase = 2 * (nb_source - n) * np.pi / nb_source
        # phase = 2 * n * np.pi / nb_source
        # pos = poss[n]
        # phase = phases[n]
        if t == 0:
            print(pos, phase)
        ax.plot(pos[1], pos[0], 'ko', mfc='none')
        p = pressure_field(t, om, q0, pos, phase)
        prt += p

    pr_time.append(prt)

# Initialisation de l'animation
pcolorm = ax.pcolormesh(TT, RR, pr_time[0].real, cmap='RdBu_r')
fig.colorbar(pcolorm, ax=ax)
ax.set_title("t = {:.4f} s".format(time[0]))
ax.set_rticks([0, R/4, R/2, 3*R/4, R])
# ax.set_rlabel_position(-np.pi/2)
ax.set_xticks(np.pi/180. * np.arange(45, 316, 90))
ax.set_theta_direction(-1)
ax.set_theta_zero_location("N") # Zero on top (north)
#ax.grid(True)

def init():
    pcolorm.set_array([])
    return pcolorm

def animate(i):
    ax.set_title("t = {:.4f} s".format(time[i]))
    pcolorm.set_array(pr_time[i].real)
    return pcolorm

anim = animation.FuncAnimation(
        fig, 
        animate, 
        frames=len(time), 
        interval=70, 
        blit=False, 
        repeat=True
)
gifname = "{}sources_{}{}.gif".format(nb_source, M, N)
# anim.save(gifname, writer='imagemagick', fps=30)
# print(gifname)
plt.show()