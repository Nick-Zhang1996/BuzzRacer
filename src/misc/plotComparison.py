import numpy as np
import matplotlib.pyplot as plt

def addDatapoint(group, samples, laptime, collision, gif):
    group.append([samples,laptime,collision,gif])

ccmppi = []
addDatapoint(ccmppi,128, 7.2436, 282,40)
addDatapoint(ccmppi,256, 5.2618,173,43)
addDatapoint(ccmppi,512, 5.5545, 133, 41)
addDatapoint(ccmppi,1024, 4.8309, 86, 43)
addDatapoint(ccmppi,2048, 5.0109, 89, 42)
addDatapoint(ccmppi,4096, 5.2564, 107, 45)

mppi = []
addDatapoint(mppi, 128, 6.0273, 152, 34)
addDatapoint(mppi, 256, 4.8655, 118, 35)
addDatapoint(mppi, 512, 4.4836, 93, 36)
addDatapoint(mppi, 1024, 4.6891, 94, 37)
addDatapoint(mppi, 2048, 4.5782, 74, 38)
addDatapoint(mppi, 4096, 4.72, 91, 39)

ccmppi = np.array(ccmppi)
mppi = np.array(mppi)

plt.plot(ccmppi[:,0], ccmppi[:,1], label='ccmppi')
plt.plot(mppi[:,0], mppi[:,1], label= 'mppi')
plt.title("Laptime (equal injected noise, 10 laps avg)")
plt.xlabel("Samples")
plt.ylabel("Laptime (s)")
plt.legend()
plt.show()

plt.plot(ccmppi[:,0], ccmppi[:,2], label='ccmppi')
plt.plot(mppi[:,0], mppi[:,2], label= 'mppi')
plt.title("Collisions (equal injected noise, 10 laps avg)")
plt.xlabel("Samples")
plt.ylabel("Collisions")
plt.legend()
plt.show()
