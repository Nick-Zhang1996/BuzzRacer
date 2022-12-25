import matplotlib.pyplot as plt
import numpy as np
vx = np.linspace(0,3,300)
#ratio = np.arctan(  (vx-0.05)*300 )/np.pi + 0.5
ratio = 1 * (vx>0.05)
plt.plot(vx,ratio)
plt.axhline(0.95)
plt.axhline(0.05)
plt.show()
