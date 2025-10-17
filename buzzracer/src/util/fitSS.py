import numpy as np
import matplotlib.pyplot as plt
throttle_ss = [0.25,0.28,0.31]
speed_ss = [0.94,1.4,1.9]
p = np.polyfit(speed_ss,throttle_ss, deg=1)

x = np.linspace(0,2.5)
y = x * p[0] + p[1]
print(p)
plt.plot(x,y)
plt.plot(speed_ss, throttle_ss,'*')
plt.show()
