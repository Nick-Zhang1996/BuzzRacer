import numpy as np
import matplotlib.pyplot as plt

xx = np.linspace(-2*np.pi-0.1,2*np.pi+0.1)
yy = (xx+np.pi)%(2*np.pi)-np.pi
plt.plot(xx,yy)
plt.show()
