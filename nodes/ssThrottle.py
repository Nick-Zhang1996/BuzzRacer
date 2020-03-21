import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

velocity = np.array([0.3,0.6,1.0,1.2,1.5])
throttle = np.array([0.18,0.2,0.22,0.23,0.24])


coeffs = np.polyfit(velocity,throttle,1)
print(coeffs)
vv = np.linspace(0,2.5)
tt = np.polyval(coeffs,vv)
plt.plot(velocity,throttle,'*')
plt.plot(vv,tt)
plt.show()
