import numpy as np
import torch
from time import time

cuda = torch.device('cuda')
x0_vec = torch.tensor( np.random.random((1000000,5)) )
x0_vec_cuda = x0_vec.cuda()

# GPU

t0 = time()
with torch.cuda.device(0):
    b = x0_vec_cuda[:,0] * x0_vec_cuda[:,1]
    c = torch.sin(x0_vec_cuda[:,2]) + torch.cos(x0_vec_cuda[:,3])
    cost_vec_cuda = b**2 + c**2
dt_cuda = time()-t0


# CPU

t0 = time()
b = x0_vec[:,0] * x0_vec[:,1]
c = torch.sin(x0_vec[:,2]) + torch.cos(x0_vec[:,3])
cost_vec = b**2 + c**2
dt_cpu = time()-t0

print(f'cuda: {dt_cuda}')
print(torch.sum(cost_vec_cuda))
print(f'cpu: {dt_cpu}')
print(torch.sum(cost_vec))

