import numpy as np
import matplotlib.pyplot as plt
v = [0.937,1.22,1.4,1.68]
T = [0.268,0.29,0.3,0.317]
p = np.polyfit(T,v,1)
print(p)
print(p[1]/p[0])


plt.plot(T,v,'*')
plt.plot(T,np.polyval(p,T))
plt.show()
