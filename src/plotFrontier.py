# analyze csv log, plot peratio frontier
import numpy as np
import matplotlib.pyplot as plt
filename = "frontier.txt"

mppi_injected = []
mppi_cov = []
ccmppi = []
with open(filename, 'r') as f:
    for line in f:
        if(line[0] == '#'):
            continue
        entry = line.split(',')
        if (entry[-2].lstrip() == "True"):
            print("bad experiment, skipping")

        entry = entry[:-2] + entry[-1:]
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

offset = 0
ccmppi = ccmppi[offset:]
mppi_injected = mppi_injected[offset:]
mppi_cov = mppi_cov[offset:]

# circle same config
for i in range(25):
    index = i
    print(index)
    print("index= %d,ratio=%.2f log no: %d, %d, %d (ccmppi, mppi_injected, mppi_cov)"%(index,ccmppi[index,-1],ccmppi[index,7], mppi_injected[index,7], mppi_cov[index,7]))
    plt.scatter(ccmppi[index,3], ccmppi[index,2],s=80,facecolor='none', edgecolor='r',label='same setting')
    plt.scatter(mppi_injected[index,3], mppi_injected[index,2],s=80,facecolor='none', edgecolor='r')
    plt.scatter(mppi_cov[index,3], mppi_cov[index,2],s=80,facecolor='none', edgecolor='r')
            
    plt.plot(ccmppi[:,3], ccmppi[:,2],'+',label='ccmppi')
    plt.plot(mppi_injected[:,3], mppi_injected[:,2],'o', label= 'mppi_injected')
    plt.plot(mppi_cov[:,3], mppi_cov[:,2], '*',label= 'mppi_cov')
    plt.title("Laptime ")
    plt.xlabel("collision")
    plt.ylabel("Laptime (s)")
    plt.legend()
    plt.show()

