# analyze csv log, plot peratio frontier
import numpy as np
import matplotlib.pyplot as plt
from common import *
filename = "fineFrontier.txt"

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
for i in range(ccmppi.shape[0]):
    index_cc = i
    ratio = ccmppi[index_cc,-1]
    print("index %d, ratio %.2f"%(index_cc, ratio))
    # search for index
    index_mppi_injected = -1
    for j in range(mppi_injected.shape[0]):
        if (np.isclose(ratio,mppi_injected[j,-1])):
            index_mppi_injected = j
            break
    if (index_mppi_injected == -1):
        print_error(" can't find matching ratio in MPPI_injected records")

    index_mppi_cov = -1
    for j in range(mppi_cov.shape[0]):
        if (np.isclose(ratio,mppi_cov[j,-1])):
            index_mppi_cov = j
            break
    if (index_mppi_cov == -1):
        print_error(" can't find matching ratio in MPPI_cov records")


    print("index= %d,ratio=%.2f log no: %d, %d, %d (ccmppi, mppi_injected, mppi_cov)"%(index_cc,ccmppi[index_cc,-1],ccmppi[index_cc,7], mppi_injected[index_mppi_injected,7], mppi_cov[index_mppi_cov,7]))
    plt.scatter(ccmppi[index_cc,3], ccmppi[index_cc,2],s=80,facecolor='none', edgecolor='r',label='same setting')
    plt.scatter(mppi_injected[index_mppi_injected,3], mppi_injected[index_mppi_injected,2],s=80,facecolor='none', edgecolor='r')
    plt.scatter(mppi_cov[index_mppi_cov,3], mppi_cov[index_mppi_cov,2],s=80,facecolor='none', edgecolor='r')
            
    plt.plot(ccmppi[:,3], ccmppi[:,2],'+',label='ccmppi')
    plt.plot(mppi_injected[:,3], mppi_injected[:,2],'o', label= 'mppi_injected')
    plt.plot(mppi_cov[:,3], mppi_cov[:,2], '*',label= 'mppi_cov')
    plt.title("Laptime ")
    plt.xlabel("collision")
    plt.ylabel("Laptime (s)")
    plt.legend()
    plt.show()

