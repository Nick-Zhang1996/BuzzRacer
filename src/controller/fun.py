# visualize some functions for cost function design
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-0.30,0.30)
y = np.arctan(-(x-0.07)*100)/np.pi*2 + 1
plt.plot(x*1e2,y,label='distance')
plt.plot([x[0]*1e2,x[-1]*1e2],[0.1,0.1])
plt.legend()
plt.show()
