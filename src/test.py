import numpy as np
import matplotlib.pyplot as plt
from math import radians

xx = np.linspace(0,radians(27))
RR = 0.102/np.tan(xx)
bb = np.arctan(0.036/RR)
yy = np.sin(bb)
plt.plot(xx,yy)
plt.show()
