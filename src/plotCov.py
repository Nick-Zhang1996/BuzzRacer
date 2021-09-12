# analyze terminal covariance
import numpy as np
import matplotlib.pyplot as plt
from common import *
import pickle
filename = "log.txt"
#filename = "fixed_fine_grid_search.txt"

def getCov(log_no):
    filename = "../log/kinematics_results/debug_dict%d.p"%(log_no)
    with open(filename,'rb') as f:
        data = pickle.load(f)
        data = data[0]

    norm = np.mean(data['pos_2_norm_vec'])
    norm = np.mean(data['state_2_norm_vec'])
    norm = np.mean(data['theory_cov_mtx_vec'])
    #norm = np.mean(data['pos_area_vec'])
    return norm


mppi_injected = []
#mppi_cov = []
ccmppi = []
with open(filename, 'r') as f:
    for line in f:
        if(line[0] == '#'):
            continue
        entry = line.split(',')
        if (entry[9].lstrip() != "False"):
            # 0->false
            entry[9] = 0
        else:
            entry[9] = 1

        if (entry[0].lstrip() == 'ccmppi'):
            ccmppi.append([float(val) for val in entry[1:]])
        if (entry[0].lstrip() == 'mppi-same-injected'):
            mppi_injected.append([float(val) for val in entry[1:]])
        '''
        if (entry[0].lstrip() == 'mppi-same-terminal-cov'):
            mppi_cov.append([float(val) for val in entry[1:]])
        '''

# algorithm, samples, car_total_laps, laptime_mean(s),  collision_count
ccmppi = np.array(ccmppi)
mppi_injected = np.array(mppi_injected)
#mppi_cov = np.array(mppi_cov)

# show terminal covariance in position as in log.txt
'''
ccmppi_pos_cov = np.mean(ccmppi[:,5])
mppi_injected_pos_cov = np.mean(mppi_injected[:,5])
mppi_cov_pos_cov = np.mean(mppi_cov[:,5])
print("position cov")
print("mppi1:" + str(mppi_injected_pos_cov))
print("mppi2:" + str(mppi_cov_pos_cov))
print("ccmppi:" + str(ccmppi_pos_cov))
'''

ccmppi_log_no = ccmppi[:,7]
mppi_injected_log_no = mppi_injected[:,7]
#mppi_cov_log_no = mppi_cov[:,7]

mppi_injected_ratio = []
#mppi_cov_ratio = []

mppi_injected_cov_vec = []
mppi_cov_cov_vec = []
ccmppi_cov_vec = []
for i in range(ccmppi.shape[0]):
    #if (ccmppi[i,8] < 0.01 or mppi_injected[i,8] < 0.01 or mppi_cov[i,8] < 0.01):  
    if (ccmppi[i,8] < 0.01 or mppi_injected[i,8] < 0.01):  
        #print("skip")
        continue
    ccmppi_cov = getCov(ccmppi_log_no[i])
    mppi_injected_cov = getCov(mppi_injected_log_no[i])
    #mppi_cov_cov = getCov(mppi_cov_log_no[i])

    mppi_injected_ratio.append(mppi_injected_cov/ccmppi_cov)
    #mppi_cov_ratio.append(mppi_cov_cov/ccmppi_cov)

    mppi_injected_cov_vec.append(mppi_injected_cov)
    #mppi_cov_cov_vec.append(mppi_cov_cov)
    ccmppi_cov_vec.append(ccmppi_cov)

print("position ellipse area (1sigma)")
print("mppi1:"+str(np.mean(mppi_injected_cov_vec)))
#print("mppi2:"+str(np.mean(mppi_cov_cov_vec)))
print("ccmppi:"+str(np.mean(ccmppi_cov_vec)))
plt.plot(mppi_injected_ratio, label='mppi 1 / ccmppi')
#plt.plot(mppi_cov_ratio, label='mppi2 / ccmppi')
plt.legend()
plt.show()

