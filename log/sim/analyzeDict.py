# analyze debug_dict
import pickle
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
sys.path.append(os.path.abspath('../../src/'))
from common import *


filename = "./debug_dict1.p"
with open(filename, 'rb') as f:
    data = pickle.load(f)
clean_mpc_crosstrack_e = np.array(data['crosstrack_error'])
clean_mpc_heading_e = np.array(data['heading_error'])

filename = "./debug_dict2.p"
with open(filename, 'rb') as f:
    data = pickle.load(f)
clean_mppi_crosstrack_e = np.array(data['crosstrack_error'])
clean_mppi_heading_e = np.array(data['heading_error'])

filename = "./debug_dict4.p"
with open(filename, 'rb') as f:
    data = pickle.load(f)
noisy_mpc_crosstrack_e = np.array(data['crosstrack_error'])
noisy_mpc_heading_e = np.array(data['heading_error'])

filename = "./debug_dict3.p"
with open(filename, 'rb') as f:
    data = pickle.load(f)
noisy_mppi_crosstrack_e = np.array(data['crosstrack_error'])
noisy_mppi_heading_e = np.array(data['heading_error'])


tt1 = np.linspace(0,clean_mpc_heading_e.shape[0],clean_mpc_heading_e.shape[0])
tt2 = np.linspace(0,clean_mpc_heading_e.shape[0],clean_mppi_heading_e.shape[0])

tt3 = np.linspace(0,noisy_mpc_heading_e.shape[0],noisy_mpc_heading_e.shape[0])
tt4 = np.linspace(0,noisy_mpc_heading_e.shape[0],noisy_mppi_heading_e.shape[0])

print("no-noise")
print("MPC %.2f"%(np.sum(clean_mpc_heading_e**2)))
print("MPPI %.2f"%(np.sum(clean_mppi_heading_e**2)))
fig = plt.figure()
ax = fig.gca()
ax.plot(tt1,np.abs(clean_mpc_heading_e),'--',label="no-noise mpc heading err")
ax.plot(tt2,np.abs(clean_mppi_heading_e),label="no-noise mppi heading err")
ax.legend()
plt.show()

print("MPC %.2f"%(np.sum(noisy_mpc_heading_e**2)))
print("MPPI %.2f"%(np.sum(noisy_mppi_heading_e**2)))
fig = plt.figure()
ax = fig.gca()
ax.plot(tt3,np.abs(noisy_mpc_heading_e),'--',label="high noise mpc heading err")
ax.plot(tt4,np.abs(noisy_mppi_heading_e),label="high noise mppi heading err")
ax.legend()
plt.show()

print("no-noise crosstrack")
print("MPC %.2f"%(np.sum(clean_mpc_crosstrack_e**2)))
print("MPPI %.2f"%(np.sum(clean_mppi_crosstrack_e**2)))
fig = plt.figure()
ax = fig.gca()
ax.plot(tt1,np.abs(clean_mpc_crosstrack_e),'--',label="no-noise mpc crosstrack err")
ax.plot(tt2,np.abs(clean_mppi_crosstrack_e),label="no-noise mppi crosstrack err")
ax.legend()
plt.show()

print("MPC %.2f"%(np.sum(noisy_mpc_crosstrack_e**2)))
print("MPPI %.2f"%(np.sum(noisy_mppi_crosstrack_e**2)))
fig = plt.figure()
ax = fig.gca()
ax.plot(tt3,np.abs(noisy_mpc_crosstrack_e),'--',label="high noise mpc crosstrack err")
ax.plot(tt4,np.abs(noisy_mppi_crosstrack_e),label="high noise mppi crosstrack err")
ax.legend()
plt.show()
