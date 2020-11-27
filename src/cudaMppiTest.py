# test mppi.cu
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import numpy as np
from time import sleep

sample_count = 10
horizon_steps = 20
control_dim = 2
state_dim = 4

f = open("./mppi.cu","r")
code = f.read()
mod = SourceModule(code)

evaluate_control_sequence = mod.get_function("evaluate_control_sequence")

#u1 = np.linspace(0,1,sample_count).astype(np.float32)
#u2 = np.linspace(1,2,sample_count).astype(np.float32)
u1 = np.ones([sample_count,horizon_steps,1])
u2 = 3*np.ones([sample_count,horizon_steps,1])
control = np.dstack([u1,u2])

# should all be 1,3
# different controls
'''
print(control[0,0,0])
print(control[0,0,1])

#different timesteps
print(control[0,5,0])
print(control[0,5,1])

#different instances
print(control[2,5,0])
print(control[2,5,1])
'''

control = control.astype(np.float32)
x0 = np.array([0,0,0,0]).astype(np.float32)
epsilon = np.zeros([sample_count,horizon_steps,control_dim]).astype(np.float32)

cost = np.zeros([sample_count]).astype(np.float32)

'''
cost = cost.flatten()
x0 = x0.flatten()
epsilon = epsilon.flatten()
'''
control = control.flatten()
print(cost.shape)
print(x0.shape)
print(control.shape)
print(epsilon.shape)
evaluate_control_sequence( 
        drv.Out(cost),drv.In(x0),drv.In(control), drv.In(epsilon),
        block=(64,1,1), grid=(1,1))
sleep(1)

# should be same 
'''
print(cost[0])
print(cost[5])
'''

# evaluate manually
def dynamics(state,control,dt):
    m1 = 1
    m2 = 1
    k1 = 1
    k2 = 1
    c1 = 1.4
    c2 = 1.4
    u = control

    x1 = state[0]
    dx1 = state[1]
    x2 = state[2]
    dx2 = state[3]

    ddx1 = -(k1*x1 + c1*dx1 + k2*(x1-x2) + c2*(dx1-dx2)-u[0])/m1
    ddx2 = -(k2*(x2-x1) + c2*(dx2-dx1)-u[1])/m2

    x1 += dx1*dt
    dx1 += ddx1*dt
    x2 += dx2*dt
    dx2 += ddx2*dt

    state[0] = x1
    state[1] = dx1
    state[2] = x2
    state[3] = dx2

    return state

def getcost(state):
    R = np.diag([1,0.1,1,0.1])
    R = R**2
    x = np.array(state) - np.array([1,0,3,0])
    return x.T @ R @ x

for i in range(sample_count):
    print(cost[i])
s = 0
x = x0.copy()
for i in range(horizon_steps):
    x = dynamics(x,[1,3],0.1)
    s += getcost(x)
print(s)



