import glob
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from hybridSim import hybridSim
from sysidDataloader import CarDataset

class MyDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]


def train(log_names):
    epochs = 20
    batch_size = 128
    torch.set_num_threads(1)
    dt = 0.01
    history_steps = 5
    forward_steps = 5

    dataset = CarDataset(log_names,dt,history_steps,forward_steps)

    dtype = torch.double
    device = torch.device('cpu') # cpu or cuda

    full_dataset = deepcopy(dataset.dataset)

    # TODO shuffle before splitting
    num_test = len(full_dataset) // 10
    #train_set = MyDataset(full_dataset[:-num_test])
    #test_set = MyDataset(full_dataset[-num_test:])

    train_set = MyDataset(full_dataset[num_test:])
    test_set = MyDataset(full_dataset[:num_test])

    train_data_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=1)
    test_data_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=1)

    criterion = nn.MSELoss()

    simulator = hybridSim(dtype, device, history_steps, forward_steps, dt)
    '''
    # FIXME debug
    for name, param in simulator.named_parameters():
        if param.requires_grad:
            print(name, param.data)
        else:
            print("no-grad",name,param.data)
    exit(0)
    '''
    simulator.to(device)
    optimizer = optim.Adam(simulator.parameters(), lr=1e-4) #default lr=1e-3

    full_states_mean = torch.tensor(dataset.full_states_mean, dtype=dtype, device=device, requires_grad=False).view(1, simulator.state_dim+simulator.action_dim)
    full_states_std = torch.tensor(dataset.full_states_std, dtype=dtype, device=device, requires_grad=False).view(1, simulator.state_dim+simulator.action_dim)

    param_history = []
    training_loss_history = []
    test_loss_history = []
    print("-------- initial values -------")
    print("mass = %.3f"%(simulator.m.detach().item()))
    print("Caf = %.3f"%(simulator.Caf.detach().item()))
    print("Car = %.3f"%(simulator.Car.detach().item()))
    print("lf = %.3f"%(simulator.lf.detach().item()))
    print("lr = %.3f"%(simulator.lr.detach().item()))
    print("Iz = %.6f"%(simulator.Iz.detach().item()))
    print("-------- -------------- -------")

    for epoch_count in range(epochs):
        #training
        epoch_loss = 0
        for i, batch in enumerate(train_data_loader):
            full_states = batch[:,:history_steps,:]
            actions = batch[:,-forward_steps:,-simulator.action_dim:]
            full_states = full_states.to(device)
            actions = actions.to(device)
            predicted_state = simulator(full_states,actions)

            target_states = batch[:,-forward_steps:,:]

            loss = criterion((predicted_state - full_states_mean) / full_states_std, (target_states - full_states_mean) / full_states_std)
            epoch_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_loss = epoch_loss.detach().item()/len(train_data_loader)
        training_loss_history.append(epoch_loss)
        print("training loss = %.4f"%(epoch_loss))

        #test
        epoch_loss = 0
        with torch.no_grad():
            for i, batch in enumerate(test_data_loader):
                full_states = batch[:,:history_steps,:]
                actions = batch[:,-forward_steps:,-simulator.action_dim:]
                full_states = full_states.to(device)
                actions = actions.to(device)
                predicted_state = simulator(full_states,actions)

                target_states = batch[:,-forward_steps:,:]

                loss = criterion((predicted_state - full_states_mean) / full_states_std, (target_states - full_states_mean) / full_states_std)
                epoch_loss += loss

        epoch_loss = epoch_loss.detach().item()/len(test_data_loader)
        test_loss_history.append(epoch_loss)
        print("test loss = %.4f"%(epoch_loss))

        mass = simulator.m.detach().item()
        Caf = simulator.Caf.detach().item()
        Car = simulator.Car.detach().item()
        lf = simulator.lf.detach().item()
        lr = simulator.lr.detach().item()
        Iz = simulator.Iz.detach().item()
        param_history.append([mass,Caf,Car,lf,lr,Iz])


    print("-------- trained values -------")
    print("mass = %.3f"%(simulator.m.detach().item()))
    print("Caf = %.3f"%(simulator.Caf.detach().item()))
    print("Car = %.3f"%(simulator.Car.detach().item()))
    print("lf = %.3f"%(simulator.lf.detach().item()))
    print("lr = %.3f"%(simulator.lr.detach().item()))
    print("Iz = %.6f"%(simulator.Iz.detach().item()))
    print("-------- -------------- -------")

    '''
    param_history = np.array(param_history)
    print("mass")
    plt.plot(param_history[:,0])
    plt.show()

    print("Ca")
    plt.plot(param_history[:,1])
    plt.plot(param_history[:,2])
    plt.show()

    print("l")
    plt.plot(param_history[:,3])
    plt.plot(param_history[:,4])
    plt.show()
    
    print("Iz")
    plt.plot(param_history[:,5])
    plt.show()
    '''

    plt.plot(training_loss_history,'r-')
    plt.plot(test_loss_history,'b.-')
    plt.show()




if __name__ == '__main__':
    #log_names =  glob.glob('../log/sysid/full_state*.p')
    log_names =  glob.glob('../log/oct9/full_state*.p')
    train(log_names)

