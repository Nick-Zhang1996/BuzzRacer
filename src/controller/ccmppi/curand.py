import os
import sys
import numpy as np
from time import sleep,time
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base_dir)

from timeUtil import execution_timer
import matplotlib.pyplot as plt

import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule

from pycuda import gpuarray

cuda_filename = "./mppi_racecar.cu"
with open(cuda_filename,"r") as f:
    code = f.read()

# prepare constants
K = 8192
T = 30
m = 2
state_dim = 6
temperature = 1
dt = 0.03

cuda_code_macros = {"SAMPLE_COUNT":K, "HORIZON":T, "CONTROL_DIM":m,"STATE_DIM":state_dim,"RACELINE_LEN":1024,"TEMPERATURE":temperature,"DT":dt}
cuda_code_macros = cuda_code_macros | {"CURAND_KERNEL_N":1024}

mod = SourceModule(code % cuda_code_macros, no_extern_c=True)

cuda_init_curand_kernel = mod.get_function("init_curand_kernel")
cuda_generate_random_var = mod.get_function("generate_normal_with_limits")

seed = np.int32(int(time()*10000))
cuda_init_curand_kernel(seed,block=(1024,1,1),grid=(1,1,1))

#device_rand_vals = gpuarray.zeros(K*T*m, dtype=np.float32)
rand_vals = np.zeros(K*T*m, dtype=np.float32)
device_rand_vals = drv.to_device(rand_vals)

# assemble limits
limits = np.array([-1,1,-2,2],dtype=np.float32)
device_limits = drv.to_device(limits)

scales = np.array([1,1],dtype=np.float32)
device_scales = drv.to_device(scales)

tic = time()
for i in range(100):
    cuda_generate_random_var(device_rand_vals,device_scales,device_limits,block=(1024,1,1),grid=(1,1,1))
    # TODO make sure rand_vals is usable by device
    rand_vals = drv.from_device(device_rand_vals,shape=(K*T*m,), dtype=np.float32)
tac = time()
t_cuda = (tac-tic)/100.0
print("cuda %.5f"%(1.0/t_cuda))
print(np.min(rand_vals[0::2]))
assert np.min(rand_vals[0::2]) > -1.01
assert np.max(rand_vals[0::2]) < 1.01

assert np.min(rand_vals[1::2]) > -2.01
assert np.max(rand_vals[1::2]) < 2.01

plt.hist(rand_vals,bins=50)
plt.show()

tic = time()
for i in range(100):
    cuda_generate_random_var(device_rand_vals,device_limits,block=(1024,1,1),grid=(1,1,1))
    # TODO make sure rand_vals is usable by device
    rand_vals = np.random.normal(size=(K,T,m))
tac = time()
t_cpu = (tac-tic)/100.0
print("cpu %.5f"%(1.0/t_cpu))
print(t_cpu/t_cuda)
