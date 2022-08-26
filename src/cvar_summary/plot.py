import matplotlib.pyplot as plt
import numpy as np

noise = [0.05,0.1,0.2,0.3,0.4,0.5]
baseline_collision = [223,190,251,201,280,214]
cvar_collision = [3,13,3,37,90,103]

plt.plot(noise,baseline_collision, label='baseline')
plt.plot(noise,cvar_collision, label='cvar')
plt.xlabel('noise')
plt.ylabel('collision')
plt.legend()
plt.show()
