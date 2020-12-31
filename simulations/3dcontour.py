import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import scipy.special as sp
from tqdm import tqdm
import numpy as np

fig, ax = plt.subplots(
    1, 1,
    subplot_kw={'projection': '3d'},
    tight_layout=False,
    figsize=(7.1, 5)
)

def kronecker(i, j):
    return 1 if i == j else 0

pts = 100
R = 1
d = 3
M = 1; N = 1; L = 1 # modes to observe in the cavity
radius = np.linspace(0, R, pts)
theta = np.linspace(0, 2 * np.pi, pts)
height = np.linspace(0, d, pts)
TT, Z = np.meshgrid(theta, height)

def amps(m, n, l, pos, om, q0):
    r0, a0, z0 = pos
    dm0 = kronecker(m, 0)
    dl0 = kronecker(l, 0)
    kz = l * np.pi / d
    kmn = sp.jnp_zeros(m, n + 1)[-1] / R 
    qstar = om * q0
    
    bst = sp.jv(m, kmn * r0) / sp.jv(m, kmn * R)**2
    #print(bst, kmn)
    amn = (2 * qstar * r0 * np.cos(m * a0) * kmn**2 * np.sin(kz * z0) * (2 - dl0)) \
            / ((1 + dm0) * np.pi * d * (k**2 - kmn**2 - kz**2) * ((kmn * R)**2 - m**2)) * bst
    bmn = (2 * qstar * r0 * np.sin(m * a0) * kmn**2 * np.sin(kz * z0) * (2 - dl0)) \
            / ((1 - dm0) * np.pi * d * (k**2 - kmn**2 - kz**2) * ((kmn * R)**2 - m**2)) * bst
    return amn, bmn, kmn, kz

def pressure_field(t, om, q0, pos, phi):
    prt = np.zeros((pts, pts, pts), dtype=complex)
    for m in tqdm(range(1, M + 1)):
        for n in range(0, N + 1):
            for l in range(0, L + 1):
                amn, bmn, kmn, kz = amps(m, n, l, pos, om, q0)
                p_given_mode = (amn * np.cos(m * TT) + bmn * np.sin(m * TT)) \
                                    * sp.jv(m, kmn * radius) * np.cos(kz * Z)
                prt += p_given_mode    
    return prt * np.exp(1j * (om * t - phi))

q0 = 1e-3
c0 = 343
k_cav = (sp.jnp_zeros(M, N + 1)[-1] / R)# + 150
print(k_cav)
k = k_cav + 0.01
om = k * c0
t = .8e-3

pos1 = (R/2, np.pi/2, d/2)
p1 = pressure_field(t, om, q0, pos1, 0)
X, Y = radius * np.cos(TT), radius * np.sin(TT)

R, T, Z = np.mgrid[0:R:pts*1j, 0:2*np.pi:pts*1j, 0:d:pts*1j]
X = R * np.cos(T)
Y = R * np.sin(T)
p1 = p1.real

# Cylinder
x=np.linspace(-1, 1, 100)
z=np.linspace(-2, 2, 100)
Xc, Zc=np.meshgrid(x, z)
Yc = np.sqrt(1-Xc**2)

# Draw parameters
ax.plot_surface(Xc, Yc, Zc, color=, fc='b', alpha=0.2)
ax.plot_surface(Xc, -Yc, Zc, color=, fc='b', alpha=0.2)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.show()