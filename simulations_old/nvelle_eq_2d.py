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

fig, ax = plt.subplots(
    1, 1,
    subplot_kw={'projection': 'polar'},
    tight_layout=False,
    figsize=(7.1, 5)
)

def findnear(array, value):
    return (np.abs(np.asarray(array) - value)).argmin()

def kronecker(i, j):
    return 1 if i == j else 0

class PointSourceCyl:
    def __init__(self, pos, q0, phi0):
        self.r0, self.a0, self.z0 = pos         # Position in space of the source
        self.q0 = q0 
        
    def field(self, params, mode):
        R, d, c0, pts = params
        m, n, l = mode
        
        # dispersion relation variables
        kz = n * np.pi / d
        kmn = sp.jnp_zeros(m, n + 1)[-1] / R
        k = (kz**2 + kmn**2)**0.5
        om = k * c0
        radius = np.linspace(0, R, pts)
        theta  = np.linspace(0, 2 * np.pi, pts) 
        height = np.linspace(0, d, pts)
        values = np.zeros((pts, pts, pts), dtype=complex)
        qstar = -1j * om * self.q0
        print(qstar, kmn, k, om)
        
        for a in tqdm(range(len(theta))):
            for h in range(len(height)):
                #expr = np.zeros_like(radius, dtype=complex)
                theta_i = theta[a]
                height_i = height[h]
                #for m in range(1, M):
                    #for n in range(0, N):
                        #for l in range(0, L):
                dm0 = kronecker(m, 0)
                dl0 = kronecker(l, 0)
                bess_ratio = sp.jv(m, kmn * self.r0) / sp.jv(m, kmn * R)**2
                print(bess_ratio)
                expr = ((qstar*self.r0*np.cos(kz*self.z0)*2*kmn**2 * (2 - dl0)) \
                       / ((k**2 - kmn**2) * ((kmn*R)**2 - m**2)*np.pi*d)) \
                       * bess_ratio * ((np.cos(m*(theta_i - self.a0))) / (1+dm0) ) \
                       * sp.jv(m, kmn * radius) * np.cos(kz * height_i)
                values[:, a, h] = expr

                            #expr = np.add(expr, ((qstar*self.r0*np.cos(kz*self.z0)*2*kmn**2 * (1-dl0)) \
                            #        / ((k**2 - kmn**2) * ((kmn*R)**2 - m**2)*np.pi*d)) \
                            #        * bess_ratio * ((np.cos(m*(theta_i - self.a0))) / (2-dm0)) \
                            #        * sp.jv(m, kmn * radius) * np.cos(kz * height_i))
        psource = values[findnear(radius, self.r0), findnear(theta, self.a0), findnear(height, self.z0)].real
        return psource, values

m = 2; n = 1; l = 1             # mode to study in the cavity
pts = 100                       # Spatial resolution
R = 1                        # Radius of the cavity
d = 0.1                         # Height of the cavity     
c0 = 340                        # speed of sound in the cavity
q0 = 1e-2                       # Source mass flow rate
pos = (0.192, np.pi/2, 0)
source0 = PointSourceCyl(
            pos, 
            q0, np.pi
)

pref, p0 = source0.field(
        (R, d, c0, pts), 
        (m, n, l)
)
print(p0[:, :, 3], '\n', pref)
radius = np.linspace(0, R, pts)
theta  = np.linspace(0, 2 * np.pi, pts) 
height = np.linspace(0, d, 3)
cmesh = ax.pcolormesh(theta, radius, p0[:, :, 15].real, cmap='RdBu_r')
fig.colorbar(cmesh, ax=ax)
ax.plot(pos[1], pos[0], 'ko', mfc='none', label='source')
#ax.set_title(r"{}{}, $k_w R$ = {:.2f}".format(m, n, kmn * R))
ax.set_rlabel_position(270)
#ax.set_rticks([0, max(radius)/2, max(radius)]); ax.set_rticklabels([])
ax.set_theta_direction(-1)
ax.set_xticks(np.pi/180. * np.arange(45, 316, 90))
ax.set_theta_zero_location("N") # Zero on top (north)
ax.grid(True)

plt.show()
