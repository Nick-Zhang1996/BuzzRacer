# visualize some functions for cost function design
import matplotlib.pyplot as plt
import numpy as np

# car-to-car collision cost
x = np.linspace(-0.30,0.30)
y = np.arctan(-(x-0.10)*100)/np.pi*2 + 1
plt.plot(x*1e2,y)
plt.plot([x[0]*1e2,x[-1]*1e2],[0,0])
plt.xlabel('distance-to-opponent')
plt.legend()
plt.show()

# car-to-boundary collision cost
# track width: 0.5-0.6
x = np.linspace(-0.30,0.30)
y = 20*(np.arctan(-(0.25-(np.abs(x)+0.05))*100)/np.pi*2+1.0)
plt.plot(x*1e2,y)
plt.title('cost map on open track')
plt.xlabel('offset-from-centerline(cm)')
plt.ylabel('cost')
plt.legend()
plt.show()

# combined cost, car on track
x = np.linspace(-0.30,0.30)
# boundary cost
y = 20*(np.arctan(-(0.25-(np.abs(x)+0.05))*100)/np.pi*2+1.0)
y[y<0] = 0
# car cost
car_pos = 0.0
y2 = (np.arctan(-(np.abs(x-car_pos)-0.10)*100)/np.pi*2 + 1)
y2[y2<0] = 0
y = y+y2
plt.plot(x*1e2,y)
plt.title('cost map on occupied track')
plt.xlabel('offset-from-centerline(cm)')
plt.ylabel('cost')
plt.legend()
plt.show()
