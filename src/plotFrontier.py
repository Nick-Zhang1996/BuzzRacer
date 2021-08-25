# analyze csv log, plot peratio frontier
import numpy as np
import matplotlib.pyplot as plt
filename = "log.txt"

mppi_injected = []
mppi_cov = []
ccmppi = []
with open(filename, 'r') as f:
    for line in f:
        if(line[0] == '#'):
            continue
        entry = line.split(',')
        if (entry[0].lstrip() == 'ccmppi'):
            ccmppi.append([float(val) for val in entry[1:]])
        if (entry[0].lstrip() == 'mppi-same-injected'):
            mppi_injected.append([float(val) for val in entry[1:]])
        if (entry[0].lstrip() == 'mppi-same-terminal-cov'):
            mppi_cov.append([float(val) for val in entry[1:]])

# algorithm, samples, car_total_laps, laptime_mean(s),  collision_count
ccmppi = np.array(ccmppi)
mppi_injected = np.array(mppi_injected)
mppi_cov = np.array(mppi_cov)
        
plt.plot(ccmppi[:,3], ccmppi[:,2],'+',label='ccmppi')
plt.plot(mppi_injected[:,3], mppi_injected[:,2],'o', label= 'mppi_injected')
plt.plot(mppi_cov[:,3], mppi_cov[:,2], '*',label= 'mppi_cov')
plt.title("Laptime ")
plt.xlabel("collision")
plt.ylabel("Laptime (s)")
plt.legend()
plt.show()

