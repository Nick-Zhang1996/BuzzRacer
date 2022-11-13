# test cpu vs gpu speed
import torch
from time import time
import numpy as np

n = int(1e7)
val1 = torch.rand(n).cpu()
val2 = torch.rand(n).cpu()
val3 = torch.rand(n).cpu()
print(f'type: {type(val1)}')

# cpu
t0 = time()
val = val1 + val2 * torch.sin(val1)*torch.cos(val2)**val1 + val3
print(f'val.device: {val.device}')
cpu_t = time()-t0
print(f'cpu used {cpu_t}')
print(np.sum(val.detach().numpy()))


# gpu
val1 = val1.cuda()
val2 = val2.cuda()
val3 = val3.cuda()

t0 = time()
val = val1 + val2 * torch.sin(val1)*torch.cos(val2)**val1 + val3
gpu_t = time()-t0
print(f'val.device: {val.device}')
print(f'gpu used {gpu_t}')

print(np.sum(val.cpu().numpy()))




