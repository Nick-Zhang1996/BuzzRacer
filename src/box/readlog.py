# analyze csv log
import numpy as np
import matplotlib.pyplot as plt
filename = "log.txt"
#filename = "20laps.txt"

mppi = []
ccmppi = []
with open(filename, 'r') as f:
    for line in f:
        if(line[0] == '#'):
            continue
        entry = line.split(',')
        if (entry[0] == 'ccmppi'):
            ccmppi.append([float(val) for val in entry[1:]])
        if (entry[0] == 'mppi'):
            mppi.append([float(val) for val in entry[1:]])

# algorithm, samples, car_total_laps, laptime_mean(s),  collision_count
ccmppi = np.array(ccmppi)
mppi = np.array(mppi)
        
plt.plot(ccmppi[:,0], ccmppi[:,2], label='ccmppi')
plt.plot(mppi[:,0], mppi[:,2], label= 'mppi')
plt.title("Laptime (equal injected noise, 100 laps avg)")
plt.xlabel("Samples")
plt.ylabel("Laptime (s)")
plt.legend()
plt.show()

plt.plot(ccmppi[:,0], ccmppi[:,3], label='ccmppi')
plt.plot(mppi[:,0], mppi[:,3], label= 'mppi')
plt.title("Collisions (equal injected noise, 100 laps avg)")
plt.xlabel("Samples")
plt.ylabel("Collisions")
plt.legend()
plt.show()


