# import os
import torch
# import tqdm
import math
import gpytorch
# from torch.nn import Linear
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution \
    # , LMCVariationalStrategy
from gpytorch.distributions import MultivariateNormal
from gpytorch.models.deep_gps import DeepGPLayer, DeepGP
from gpytorch.mlls import DeepApproximateMLL, VariationalELBO
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from matplotlib import pyplot as plt
from time import time, sleep
from gpModel import MultitaskDeepGP

from tire import tireCurve

# RUN PARAMETER

smoke_test = False


# kinematics model

# base model
# N: number of datapoints
# n: dimension of combined state
# state_control: [state, control] dim: N*n
def kinematics(state_control):
    L = 0.09
    lf = 0.04824
    lr = L - lf
    dt = 0.01

    v_forward = state_control[:, 0]
    v_sideway = state_control[:, 1]
    omega = state_control[:, 2]
    throttle = state_control[:, 3]
    steering = state_control[:, 4]

    v = v_forward

    beta = torch.arctan(torch.tan(steering) * lr / (lf+lr))
    dvdt = 6.17*(throttle - v/15.2 - 0.333)
    omega = dheadingdt = v/lr*torch.sin(beta)

    dv_sideway = 0*dvdt
    dstatedt = torch.stack([dvdt, dv_sideway, omega], -1)
    return dstatedt


def dynamics(state_control):
    '''
    given concatednated state+control
    provide state derivative
    state_control: [vx, vy, omega, throttle, steering]
    return : torch.stack([dvx,dvy,omega],-1)
    '''
    Iz = 417757e-9
    m = 0.1667

    L = 0.09
    lf = 0.04824
    lr = L - lf
    dt = 0.01

    # NOTE here vx = vf, vy = vs, different convention
    vx = v_forward = state_control[:, 0]
    vy = v_sideway = state_control[:, 1]

    omega = state_control[:, 2]
    throttle = state_control[:, 3]
    steering = state_control[:, 4]

    dvdt = 0
    dv_sideway = 0

    # for small longitudinal velocity use kinematic model
    if (False):
        beta = atan(lr/L*tan(steering))
        def norm(a, b): return (a**2+b**2)**0.5
        # motor model
        d_vx = 6.17*(throttle - vx/15.2 - 0.333)
        vx = vx + d_vx * dt
        vy = norm(vx, vy)*sin(beta)
        d_omega = 0.0
        omega = vx/L*tan(steering)

        slip_f = 0
        slip_r = 0
        Ffy = 0
        Fry = 0

    else:
        slip_f = -torch.arctan((omega*lf + vy)/vx) + steering
        slip_r = torch.arctan((omega*lr - vy)/vx)

        # Ffy = Df * np.sin( C * np.arctan(B *slip_f)) * 9.8 * lr / (lr + lf) * m
        # Fry = Dr * np.sin( C * np.arctan(B *slip_r)) * 9.8 * lf / (lr + lf) * m
        Ffy = tireCurve(slip_f) * m * 9.8 * lr/(lr+lf)
        Fry = 1.15*tireCurve(slip_r) * m * 9.8 * lf/(lr+lf)

        # Dynamics
        # d_vx = 1.0/m * (Frx - Ffy * np.sin( steering ) + m * vy * omega)
        d_vx = 6.17*(throttle - vx/15.2 - 0.333)
        d_vy = 1.0/m * (Fry + Ffy * torch.cos(steering) - m * vx * omega)
        d_omega = 1.0/Iz * (Ffy * lf * torch.cos(steering) - Fry * lr)

        omega = omega + d_omega * dt

    dvdt = d_vx
    dv_sideway = d_vy

    dstatedt = torch.stack([dvdt, dv_sideway, omega], -1)

    return dstatedt


# TRAINING DATA

'''
train_x = torch.stack([torch.rand(training_pts), torch.rand(training_pts)],-1)
train_y = torch.stack([
    torch.sin((2 * math.pi)*train_x[:,0]),
    -2.5*torch.cos((2 * math.pi)*train_x[:,1]**2)*torch.exp(-2*train_x[:,0]),
    ],-1)
'''
training_pts = 10000
# v_forward,v_sideway,omega, throttle, steering
train_x = torch.stack([
    torch.rand(training_pts)*3,
    torch.rand(training_pts)*0.3-0.15,
    torch.rand(training_pts)*0.1-0.05,
    torch.rand(training_pts)*2-1,
    ((torch.rand(training_pts)-0.5)*2)*math.radians(26.1)
], -1)
train_y = dynamics(train_x)

# TRAINING MODEL

model = MultitaskDeepGP(train_x.shape, train_y.size(-1))

model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
mll = DeepApproximateMLL(VariationalELBO(
    model.likelihood, model, num_data=train_y.size(0)))

# 200
num_epochs = 10

for i in range(num_epochs):
    optimizer.zero_grad()
    output = model(train_x)
    loss = -mll(output, train_y)
    loss.backward()
    optimizer.step()
    print(i)

# TESTING ON DATA WHERE x1 = 0-1, x2 = 0.5

'''
test_x = torch.stack([torch.linspace(0, 1, 25), torch.ones(25)*0.5],-1)   
test_y = torch.stack([
    torch.sin((2 * math.pi)*test_x[:,0]),
    -2.5*torch.cos((2 * math.pi)*test_x[:,1]**2)*torch.exp(-2*test_x[:,0]),
    ],-1)
'''
# v_forward,v_sideway,omega, throttle, steering
test_x = torch.stack([
    torch.ones(100)*1.0,
    torch.zeros(100),
    torch.zeros(100),
    torch.zeros(100),
    torch.linspace(-1, 1, 100)*math.radians(26),
], -1)
test_y = dynamics(test_x)

model.eval()
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    mean, var = model.predict(test_x)
    lower = mean - 2 * var.sqrt()
    upper = mean + 2 * var.sqrt()

# Plot results
fig, ax = plt.subplots(figsize=(8, 6), dpi=240)

plt.subplot(2, 2, 1)
plt.title('speed = 1.0, dv_sideway vs steering')
plt.plot(test_x[:, 4].numpy(), test_y[:, 1].numpy(), ':k')
plt.plot(test_x[:, 4].numpy(), mean[:, 1].numpy(), 'k')
plt.fill_between(test_x[:, 4].numpy(), lower[:, 1].numpy(),
                 upper[:, 1].numpy(), facecolor='k', alpha=0.2)
plt.xlabel('x1')
plt.ylabel('y1')
plt.legend(['True', 'Predicted', '2 Sigma'])

plt.subplot(2, 2, 2)
plt.title('placeholder, ')
plt.plot(test_x[:, 3].numpy(), test_y[:, 0].numpy(), ':k')
plt.plot(test_x[:, 3].numpy(), mean[:, 0].numpy(), 'k')
plt.fill_between(test_x[:, 3].numpy(), lower[:, 0].numpy(),
                 upper[:, 0].numpy(), facecolor='k', alpha=0.2)
plt.xlabel('x1')
plt.ylabel('y2')
plt.legend(['True', 'Predicted', '2 Sigma'])

# v_forward,v_sideway,omega, throttle, steering
test_x = torch.stack([
    torch.linspace(0, 3, 100),
    torch.zeros(100),
    torch.zeros(100),
    torch.zeros(100),
    torch.ones(100)*math.radians(10),
], -1)
test_y = dynamics(test_x)

t0 = time()
model.eval()
dt = time()-t0
print(f'100 examples took {dt}s')
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    mean, var = model.predict(test_x)
    lower = mean - 2 * var.sqrt()
    upper = mean + 2 * var.sqrt()

plt.subplot(2, 2, 3)
plt.title('dv_sideway vs v_forward, given steering')
plt.plot(test_x[:, 0].numpy(), test_y[:, 1].numpy(), ':k')
plt.plot(test_x[:, 0].numpy(), mean[:, 1].numpy(), 'k')
plt.fill_between(test_x[:, 0].numpy(), lower[:, 1].numpy(),
                 upper[:, 1].numpy(), facecolor='k', alpha=0.2)
plt.xlabel('x2')
plt.ylabel('y1')
plt.legend(['True', 'Predicted', '2 Sigma'])

plt.subplot(2, 2, 4)
plt.title('placeholder')
plt.plot(test_x[:, 0].numpy(), test_y[:, 0].numpy(), ':k')
plt.plot(test_x[:, 0].numpy(), mean[:, 0].numpy(), 'k')
plt.fill_between(test_x[:, 0].numpy(), lower[:, 0].numpy(),
                 upper[:, 0].numpy(), facecolor='k', alpha=0.2)
plt.xlabel('x2')
plt.ylabel('y2')
plt.legend(['True', 'Predicted', '2 Sigma'])

plt.tight_layout()
plt.show()
