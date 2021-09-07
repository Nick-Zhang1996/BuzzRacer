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
            #print("bad experiment, skipping")
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


# circle same config

for index in range(mppi_cov.shape[0]):
    index_mppi_cov = index
    index_cc = -1
    index_mppi_injected = -1

    alfa = mppi_cov[index_mppi_cov,8]
    beta = mppi_cov[index_mppi_cov,9]
    print("mppi cov index %d, alfa %.2f beta %.2f"%(index_mppi_cov, alfa, beta))
    for i in range(ccmppi.shape[0]):
        if (np.isclose(alfa,ccmppi[i,8]) and np.isclose(beta,ccmppi[i,9])): 
            index_cc = i
    if (index_cc == -1):
        print_error("can't find cc index")
    for i in range(mppi_injected.shape[0]):
        if (np.isclose(alfa,mppi_injected[i,8]) and np.isclose(beta,mppi_injected[i,9])): 
            index_mppi_injected = i
    if (index_mppi_injected == -1):
        print_error("can't find mppi injected index")
    print("index: cc: %d, mppi-cov: %d, mppi-injected: %d"%(ccmppi[index_cc,7], mppi_cov[index_mppi_cov,7], mppi_injected[index_mppi_injected,7]))


            
    plt.plot(ccmppi[:,3], ccmppi[:,2],'o',label='ccmppi')
    plt.plot(mppi_injected[:,3], mppi_injected[:,2],'o', label= 'MPPI 1')
    plt.plot(mppi_cov[:,3], mppi_cov[:,2], 'o',label= 'MPPI 2')

    plt.scatter(ccmppi[index_cc,3], ccmppi[index_cc,2],s=80,facecolor='none', edgecolor='r',label='same setting', zorder=10)
    plt.scatter(mppi_injected[index_mppi_injected,3], mppi_injected[index_mppi_injected,2],s=80,facecolor='none', edgecolor='r', zorder=10)
    plt.scatter(mppi_cov[index_mppi_cov,3], mppi_cov[index_mppi_cov,2],s=80,facecolor='none', edgecolor='r', zorder=10)
    plt.title("Laptime ")
    plt.xlabel("collision")
    plt.ylabel("Laptime (s)")
    plt.legend()
    plt.show()

