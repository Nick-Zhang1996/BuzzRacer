# testing spline
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev,CubicSpline

theta = 2*np.pi* np.linspace(0, 1, 5)
y = np.c_[np.cos(theta), np.sin(theta)]
cs = CubicSpline(theta, y, bc_type='periodic')
print("ds/dx={:.1f} ds/dy={:.1f}".format(cs(0, 1)[0], cs(0, 1)[1]))
xs = 2 * np.pi * np.linspace(0, 1, 100)
plt.figure(figsize=(6.5, 4))
plt.plot(y[:, 0], y[:, 1], 'o', label='data')
plt.plot(np.cos(xs), np.sin(xs), label='true')
plt.plot(cs(xs)[:, 0], cs(xs)[:, 1], label='spline')
plt.axes().set_aspect('equal')
plt.legend(loc='center')
plt.show()
