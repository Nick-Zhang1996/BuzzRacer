import sys
import os.path
import os
import matplotlib.pyplot as plt
import torch
import numpy as np
import pickle
from gpModel import MultitaskDeepGP
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution \
    # , LMCVariationalStrategy
from gpytorch.distributions import MultivariateNormal
from gpytorch.models.deep_gps import DeepGPLayer, DeepGP
from gpytorch.mlls import DeepApproximateMLL, VariationalELBO
from gpytorch.likelihoods import MultitaskGaussianLikelihood

def loadData(filename='log/2023_9_25_exp/full_state2.p', visualize=False):
    '''
    Data Preparation. given full_state.p log, provide:
    output_data_x: [vx,vy, omega, throttle, steering] dim= (N,5)
    output_data_y: [d_vx, d_vy, omega]
    '''
    basedir = os.path.abspath(__file__)
    for i in range(4):
        basedir = os.path.dirname(basedir)
    full_path = os.path.join(basedir,filename)
    print(full_path)

    # data dimension: timestep, cars, state
    with open(full_path,'rb') as f:
        data = np.array(pickle.load(f),dtype=np.float32)
    car_count = data.shape[1]

    # time, x,y,heading,v_forward,v_sideway,omega, steering, throttle
    start = 920
    end = 2200
    data[:,:,0] -= data[0,0,0]

    output_data_x = torch.from_numpy(data[start:end,0,4:])
    # TODO noise reduction
    # TODO use actual time diff
    d_vx = np.diff(data[:,0,4])[start:end]/0.01
    d_vy = np.diff(data[:,0,5])[start:end]/0.01
    omega = data[start+1:end+1,0,6]
    output_data_y = torch.from_numpy(np.stack([d_vx,d_vy,omega],-1))


    if (visualize):
        plt.plot(data[start:end,0,0], data[start:end,0,4])
        plt.plot(data[start:end,0,0], data[start:end,0,8])
        plt.plot(data[start:end,0,0], data[start:end,0,7])
        plt.legend(['vx','T','S'])
        plt.show()
    return (output_data_x, output_data_y)

def train(model,train_x,train_y):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    mll = DeepApproximateMLL(VariationalELBO(model.likelihood, model, num_data=train_y.size(0)))

    num_epochs = 100

    for i in range(num_epochs):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()
        print(f'ep {i}, loss = {loss.item()}')

def test(model, test_x, test_y):
        
    ## TESTING ON DATA WHERE x1 = 0-1, x2 = 0.5
        
    # v_forward,v_sideway,omega, throttle, steering
    test_x = torch.stack([
        torch.ones(100)*1.0,
        torch.zeros(100),
        torch.zeros(100),
        torch.zeros(100),
        torch.linspace(-1,1,100)*math.radians(26),
        ],-1)
    test_y = dynamics(test_x)
        
    model.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        mean, var = model.predict(test_x)
        lower = mean - 2 * var.sqrt()
        upper = mean + 2 * var.sqrt()


if __name__=='__main__':
    train_x,train_y = loadData(visualize=False)
    print(train_x.shape, train_y.shape)

    ## TRAINING MODEL
    model = MultitaskDeepGP(train_x.shape, train_y.size(-1))
    train(model, train_x, train_y)

