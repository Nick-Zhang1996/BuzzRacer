# analyze csv log, plot peratio frontier
import numpy as np
import matplotlib.pyplot as plt
from common import *
filename = "log.txt"
#filename = "successful_frontier.txt"

mppi_injected = []
mppi_cov = []
ccmppi = []
with open(filename, 'r') as f:
    for line in f:
        if(line[0] == '#'):
            continue
        entry = line.split(',')
        if (entry[9].lstrip() != "False"):
            print("bad experiment, skipping")
            continue

        entry = entry[:9] + entry[10:]
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
#alfa 8
#beta 9
'''
mask = ccmppi[:,8] < 1.2
ccmppi = ccmppi[mask]
mask = mppi_injected[:,8] < 1.2
mppi_injected = mppi_injected[mask]
mppi_cov = mppi_cov[offset:]
'''


# circle same config
for i in range(1):
    index_cc = i
    alfa = ccmppi[index_cc,8]
    beta = ccmppi[index_cc,9]

    print("index %d, alfa %.2f beta %.2f"%(index_cc, alfa, beta))
    #print("index= %d,ratio=%.2f log no: %d, %d, %d (ccmppi, mppi_injected, mppi_cov)"%(index_cc,ccmppi[index_cc,-1],ccmppi[index_cc,7], mppi_injected[index_mppi_injected,7], mppi_cov[index_mppi_cov,7]))
    #print("index= %d,ratio=%.2f log no: %d, %d (ccmppi, mppi_injected)"%(index_cc,ccmppi[index_cc,-1],ccmppi[index_cc,7], mppi_injected[index_mppi_injected,7]))
    #plt.scatter(ccmppi[index_cc,3], ccmppi[index_cc,2],s=80,facecolor='none', edgecolor='r',label='same setting')
    #plt.scatter(mppi_injected[index_mppi_injected,3], mppi_injected[index_mppi_injected,2],s=80,facecolor='none', edgecolor='r')
    #plt.scatter(mppi_cov[index_mppi_cov,3], mppi_cov[index_mppi_cov,2],s=80,facecolor='none', edgecolor='r')
            
    plt.plot(ccmppi[:,3], ccmppi[:,2],'+',label='ccmppi')
    plt.plot(mppi_injected[:,3], mppi_injected[:,2],'o', label= 'mppi_injected')
    plt.plot(mppi_cov[:,3], mppi_cov[:,2], '*',label= 'mppi_cov')
    plt.title("Laptime ")
    plt.xlabel("collision")
    plt.ylabel("Laptime (s)")
    plt.legend()
    plt.show()

