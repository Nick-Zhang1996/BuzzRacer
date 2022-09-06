import matplotlib.pyplot as plt
import numpy as np

# plot baseline vs cvar under different noise, old experiment
'''
noise = [0.05,0.1,0.2,0.3,0.4,0.5]
baseline_collision = [223,190,251,201,280,214]
cvar_collision = [3,13,3,37,90,103]

plt.plot(noise,baseline_collision, label='baseline')
plt.plot(noise,cvar_collision, label='cvar')
plt.xlabel('noise')
plt.ylabel('collision')
plt.legend()
plt.show()
'''

# study different A, cvar_a=0.95, Cu = 0.5, grid7a
A = [2,4,6,8,10]
cvar_collision1 = np.array([1091,799,603,508,378])
cvar_collision2 = np.array([404,358,244,271,220])
cvar_collision = cvar_collision1 + cvar_collision2

plt.plot(A,cvar_collision, '*',label='cvar')
plt.xlabel('cvar_A')
plt.ylabel('collision')
plt.legend()
plt.show()
