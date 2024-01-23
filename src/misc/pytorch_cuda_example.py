import numpy as np
import torch

x0_vec = torch.tensor( np.random.random((10,5)) )
x0_vec_cuda = x0_vec.cuda()

# GPU
with torch.cuda.device(1):
    b = x0_vec_cuda[:,0] * x0_vec_cuda[:,1]
    c = torch.sin(x0_vec_cuda[:,2]) + torch.cos(x0_vec_cuda[:,3])
    cost_vec_cuda = b**2 + c**2


# CPU
    b = x0_vec[:,0] * x0_vec[:,1]
    c = torch.sin(x0_vec[:,2]) + torch.cos(x0_vec[:,3])
    cost_vec = b**2 + c**2

print(torch.sum(cost_vec_cuda))
print(torch.sum(cost_vec))

