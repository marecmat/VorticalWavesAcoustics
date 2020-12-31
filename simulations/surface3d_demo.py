from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X, Y, Z = axes3d.get_test_data(0.05)
cset = ax.contour(X, Y, Z, levels=np.linspace(-1,1,40), cmap=cm.coolwarm)
# ax.clabel(cset, fontsize=9, inline=1)

plt.show()
