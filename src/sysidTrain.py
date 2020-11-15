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
from advCarSim import advCarSim

class MyDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]

def test(test_data_loader,history_steps,forward_steps,simulator,device,criterion,optimizer,test_loss_history,err_history,enable_rnn):
    #test
    epoch_loss = 0
    cum_error = None
    with torch.no_grad():
        for i, batch in enumerate(test_data_loader):
            full_states = batch[:,:history_steps,:]
            actions = batch[:,-forward_steps:,-simulator.action_dim:]
            full_states = full_states.to(device)
            actions = actions.to(device)
            predicted_state = simulator(full_states,actions,enable_rnn)

            target_states = batch[:,-forward_steps:,:]

            #loss = criterion((predicted_state - full_states_mean) / full_states_std, (target_states - full_states_mean) / full_states_std)
            # angle wrapping
            # change target states
            target_states[:,:,4] = (target_states[:,:,4] - predicted_state[:,:,4] + np.pi)%(2*np.pi) - np.pi + predicted_state[:,:,4]

            loss = criterion(predicted_state, target_states)
            epoch_loss += loss

            # loss is difficult to understand, we now calculate the state difference
            # as fraction of error
            src = full_states[:,-1,:].detach().numpy()
            ref = target_states[:,-1,:].detach().numpy()
            test = predicted_state[:,-1,:].detach().numpy()
            diff = state_diff(src,ref,test)

            error = diff
            #print(error)
            #error = np.mean(diff)
            if cum_error is None:
                cum_error = error
            else:
                cum_error += error

            # DEBUG
            # plot states
            '''
            for p in range(100):
                print(predicted_state[p,-1,:])
                
                fig = plt.figure()
                ax = fig.gca()
                t_hist = np.linspace(0,history_steps-1,history_steps)
                t_future = np.linspace(history_steps,forward_steps+history_steps-1,forward_steps)
                for j in range(6):
                    ax.plot(t_hist,full_states[p,:,j],'r-', label="hist")
                    ax.plot(t_future,target_states[p,:,j],'r-', label="target")
                    ax.plot(t_future,predicted_state[p,:,j],'b--', label="predicted")

                #ax.legend()
                plt.show()
            '''
    
    test_loss = epoch_loss.detach().item()/len(test_data_loader)
    test_loss_history.append(test_loss)
    cum_error /= len(test_data_loader)
    err_history.append(cum_error)
    return test_loss,error

def train(train_data_loader,history_steps,forward_steps,simulator,device,criterion,optimizer,train_loss_history,err_history,enable_rnn):
        #training
        epoch_loss = 0
        cum_error = None
        for i, batch in enumerate(train_data_loader):
            full_states = batch[:,:history_steps,:]
            actions = batch[:,-forward_steps:,-simulator.action_dim:]
            full_states = full_states.to(device)
            actions = actions.to(device)

            #truth_state = simulator.testForward(full_states.clone(), actions, i=0,sim=ground_truth_sim)
            predicted_state = simulator(full_states,actions,enable_rnn)

            target_states = batch[:,-forward_steps:,:]
            # angle wrapping
            # change target states
            target_states_wrapped = target_states.clone()
            target_states_wrapped[:,:,4] = (target_states[:,:,4] - predicted_state[:,:,4] + np.pi)%(2*np.pi) - np.pi + predicted_state[:,:,4]

            #loss = criterion((predicted_state - full_states_mean) / full_states_std, (target_states - full_states_mean) / full_states_std)
            loss = criterion(predicted_state, target_states_wrapped)
            # FIXME
            #print(loss)

            latest_state = full_states[:,-1,-(simulator.state_dim+simulator.action_dim):-simulator.action_dim]
            latest_action = full_states[:,-1,-simulator.action_dim:]
            latest_full_state = np.hstack([latest_state.detach().numpy(),latest_action.detach().numpy()])

            '''
            print("predicted state, hybridsim")
            print(predicted_state.detach().numpy()-target_states.detach().numpy())
            print("predicted state, ground truth sim")
            print(truth_state-target_states.detach().numpy())
            print("target state, training set")
            print(target_states.detach().numpy()-latest_full_state)
            '''

            epoch_loss += loss.detach().item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # loss is difficult to understand, we now calculate the state difference
            # as fraction of error
            src = full_states[:,-1,:].detach().numpy()
            ref = target_states_wrapped[:,-1,:].detach().numpy()
            test = predicted_state[:,-1,:].detach().numpy()
            diff = state_diff(src,ref,test)
            error = np.mean(diff)
            if cum_error is None:
                cum_error = error
            else:
                cum_error += error

        train_loss = epoch_loss/len(train_data_loader)
        train_loss_history.append(train_loss)
        cum_error /= len(train_data_loader)
        err_history.append(cum_error)
        return train_loss


# calculate state difference
# src: starting state
# ref: reference target state
# test: state to be tested
# each state is first normalized
# criterion: (test-ref)/(ref-src), in norm
def state_diff(src,ref,test):
    #temp = np.vstack([src,ref,test])
    #state_mean = np.mean(temp,axis=0)
    #state_stddev = np.std(temp,axis=0)
    #src = (src-state_mean)/state_stddev
    #ref = (ref-state_mean)/state_stddev
    #test = (test-state_mean)/state_stddev

    #base_norm = np.linalg.norm(src-ref,axis=1)
    #diff_norm = np.linalg.norm(test-ref,axis=1)
    #return diff_norm/base_norm

    # DEBUG
    diff = np.abs(test-ref)

    # check if angle wrapping is done correctly
    angle_diff = np.max(diff[:,4])
    if angle_diff > 1.9*np.pi:
        print("angle diff too large")


    diff = np.mean(diff,axis=0)

    return diff

def angle_diff(a,b):
    diff = a-b
    return (diff+np.pi)%(2*np.pi)-np.pi


def sysid(log_names):
    epochs = 20
    batch_size = 256
    torch.set_num_threads(1)
    dt = 0.01
    history_steps = 5
    forward_steps = 3
    learning_rate = 1e-5
    enable_rnn = True

    dataset = CarDataset(log_names,dt,history_steps,forward_steps)

    dtype = torch.double
    device = torch.device('cpu') # cpu or cuda


    # shuffle before splitting
    # this may be undesirable
    np.random.shuffle(dataset.dataset)
    full_dataset = dataset.dataset
    #full_dataset = deepcopy(dataset.dataset)

    num_test = len(full_dataset) // 10
    #train_set = MyDataset(full_dataset[:-num_test])
    #test_set = MyDataset(full_dataset[-num_test:])

    # another way of splitting
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
    optimizer = optim.Adam(simulator.parameters(), lr=learning_rate) #default lr=1e-3

    full_states_mean = torch.tensor(dataset.full_states_mean, dtype=dtype, device=device, requires_grad=False).view(1, simulator.state_dim+simulator.action_dim)
    full_states_std = torch.tensor(dataset.full_states_std, dtype=dtype, device=device, requires_grad=False).view(1, simulator.state_dim+simulator.action_dim)

    param_history = []
    train_loss_history = []
    test_loss_history = []
    train_err_history = []
    test_err_history = []
    print("-------- initial values -------")
    print("mass = %.3f"%(simulator.m.detach().item()))
    print("Caf = %.3f"%(simulator.Caf().detach().item()))
    print("Car = %.3f"%(simulator.Car().detach().item()))
    print("lf = %.3f"%(simulator.lf.detach().item()))
    print("lr = %.3f"%(simulator.lr.detach().item()))
    print("Iz = %.6f"%(simulator.Iz().detach().item()))
    print("throttle offset = %.4f"%(simulator.throttle_offset.detach().item()))
    print("throttle ratio = %.4f"%(simulator.throttle_ratio.detach().item()))
    print("-------- -------------- -------")

    test_loss,error = test(test_data_loader,history_steps,forward_steps,simulator,device,criterion,optimizer,test_loss_history,test_err_history,enable_rnn)
    print("initial test cost %.5f (err = %.5f)"%(test_loss,-1))
    print(error)

    for epoch_count in range(epochs):

        train_loss = train(train_data_loader,history_steps,forward_steps,simulator,device,criterion,optimizer,train_loss_history,train_err_history,enable_rnn)

        test_loss,error = test(test_data_loader,history_steps,forward_steps,simulator,device,criterion,optimizer,test_loss_history,test_err_history,enable_rnn)

        #print("Train loss = %.6f, Test loss = %.6f (err=%.5f)"%(train_loss,test_loss,error))
        print("Train loss = %.6f, Test loss = %.6f "%(train_loss,test_loss))
        print(error)

        # log parameter update history
        mass = simulator.m.detach().item()
        Caf = simulator.Caf().detach().item()
        Car = simulator.Car().detach().item()
        lf = simulator.lf.detach().item()
        lr = simulator.lr.detach().item()
        Iz = simulator.Iz().detach().item()
        param_history.append([mass,Caf,Car,lf,lr,Iz])


    print("-------- trained values -------")
    print("mass = %.3f"%(simulator.m.detach().item()))
    print("Caf = %.3f"%(simulator.Caf().detach().item()))
    print("Car = %.3f"%(simulator.Car().detach().item()))
    print("lf = %.3f"%(simulator.lf.detach().item()))
    print("lr = %.3f"%(simulator.lr.detach().item()))
    print("Iz = %.6f"%(simulator.Iz().detach().item()))
    print("throttle offset = %.4f"%(simulator.throttle_offset.detach().item()))
    print("throttle ratio = %.4f"%(simulator.throttle_ratio.detach().item()))
    print("-------- -------------- -------")

    param_history = np.array(param_history)
    '''
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

    plt.plot(train_loss_history,'r-')
    plt.plot(test_loss_history,'b.-')
    plt.show()

    # plot acc
    fig = plt.figure()
    ax = fig.gca()
    ax.plot(train_loss_history, label="train loss")
    ax.plot(test_loss_history, label="test loss")
    ax.legend()
    plt.show()

    fig = plt.figure()
    ax = fig.gca()
    ax.plot(test_err_history, label="test err")
    ax.plot(train_err_history, label="train err")
    ax.legend()
    plt.show()




if __name__ == '__main__':
    # simulation data
    #log_names =  glob.glob('../log/sysid/full_state*.p')
    # real data
    log_names =  glob.glob('../log/nov10/full_state*.p')
    sysid(log_names)

