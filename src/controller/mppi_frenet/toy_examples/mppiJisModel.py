# Ji's model, in mppi, run on CPU python and GPU CUDA
import os
import sys
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base_dir)
from math import sin
import numpy as np
import matplotlib.pyplot as plt
from mppi import MPPI
from common import *

class Model:
    def __init__(self,x0=[10,0,1,3],x_goal=[-5,0,0,0]):
        # FIXME debug
        #x0 = [-5,0,0,0]
        self.samples_count = 1024
        self.horizon_steps = 20
        self.temperature = 1.0
        self.dt = 0.1
        noise = 0.1

        self.state_dim = 4
        self.control_dim = 2

        # ----

        self.noise_cov = np.eye(self.control_dim,dtype=np.float32)*noise

        self.A = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],dtype=np.float32)
        self.B = np.array([[0.5,0],[0,0.5],[1,0],[0,1]],dtype=np.float32)
        self.D = 0.3*np.eye(self.state_dim)

        self.Q = np.eye(self.state_dim,dtype=np.float32)
        self.R = np.eye(self.control_dim,dtype=np.float32)


        self.x = np.array(x0,dtype=np.float32)
        self.t = 0

        self.x_goal = np.array(x_goal,dtype=np.float32)

        self.t_hist = []
        self.x_hist = []
        self.u_hist = []
        self.cost_hist = []

    def plot(self):
        xx = np.array(self.x_hist)
        # should be -5,0
        plt.plot(xx[:,0],xx[:,1],'*')
        plt.show()
        cc = np.array(self.cost_hist)
        plt.plot(cc)
        plt.show()

    # step dynamics
    def step(self,u,dt=None):
        if dt is None:
            dt = self.dt
        self.x = self.calcNextState(self.x,u,dt)
        self.t += dt

        self.t_hist.append(self.t)
        self.x_hist.append(self.x)
        self.u_hist.append(u)
        return

    def calcNextState(self,state,u,dt):
        u = np.array(u)
        assert u.shape == (self.control_dim,)
        #return state + self.A @ state * dt + self.B @ u * dt
        return  self.A @ state + self.B @ u


    def getCost(self,state,u):
        u = np.array(u)
        return (state-self.x_goal).T @ self.Q @ (state-self.x_goal) + u.T @ self.R @ u

    def terminalConditionVal(self):
        return np.linalg.norm(self.x[0:2]-self.x_goal[0:2])

    def run(self):
        mppi = MPPI(self.samples_count,self.horizon_steps,self.control_dim,self.temperature,self.dt,self.noise_cov,cuda=True)
        mppi.applyDiscreteDynamics = self.calcNextState
        mppi.evaluateCost = self.getCost

        # warm start
        ref_control = [[0,0]]*self.horizon_steps
        warm_start_iter = 5
        for i in range(warm_start_iter):
            print("warm start iter = %d/%d"%(i,warm_start_iter))
            control_limit = [[-1,1]]*2
            uu = mppi.control_goal_state(self.x_goal,self.x,ref_control,control_limit)
            ref_control = np.vstack([uu[1:,:],uu[-1,:]])

        while (self.t<5.0):
            print(" sim t = %.2f"%(self.t))
            print(self.x)
            control_limit = [[-1,1]]*2
            # NOTE start from no control every time
            #ref_control = [[0,0]]*self.horizon_steps
            uu = mppi.control_goal_state(self.x_goal,self.x,ref_control,control_limit)

            '''
            # NOTE run a small simulation, show projected future if uu is faithfully carried out to the end
            sim_x_hist = []
            sim_x = self.x.copy()
            sim_x_hist.append(sim_x)
            for u in uu:
                sim_x = self.calcNextState(sim_x,u,self.dt)
                sim_x_hist.append(sim_x)

            xx = np.array(self.x_hist)
            projected_xx = np.array(sim_x_hist)
            if len(xx)>0:
                plt.plot(xx[:,0],xx[:,1],'*')
                plt.plot(projected_xx[:,0],projected_xx[:,1])
                print_info("cuda")
                plt.show()
            '''


            ref_control = np.vstack([uu[1:,:],uu[-1,:]])
            #ref_control = np.vstack([uu[1:,:],np.zeros([1,self.control_dim])])
            main.step(uu[0,:])

            terminalVal = self.terminalConditionVal()
            main.cost_hist.append(terminalVal)
            print_ok(terminalVal)

            if (terminalVal < 1.0):
                print_ok("terminal condition met")
                break
            elif (terminalVal > 20):
                print_warning("diverged")
                break

        mppi.p.summary()
        self.plot()

if __name__=="__main__":
    main = Model()
    main.run()


