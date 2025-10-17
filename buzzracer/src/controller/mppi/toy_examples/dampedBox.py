import numpy as np
import matplotlib.pyplot as plt
from mpc import MPC
from time import time

class Box:
    def __init__(self):
        self.m = m = 1
        self.k = k = 1
        self.c = c = 0.3
        self.t_hist = []
        self.x_hist = []
        self.u_hist = []
        self.t = 0
        self.x = np.array([[0.0,0.0]]).T
        self.A = np.array([[0,1],[-k/m, -c/m]])
        self.B = np.array([[0,1.0/m]]).T

        self.x_hist.append(self.x)
        self.u_hist.append(0)
        self.t_hist.append(self.t)
        return

    def step(self,u,dt):
        self.t += dt
        self.x = self.x + (self.A @ self.x + self.B * u) * dt

        self.x_hist.append(self.x)
        self.u_hist.append(np.sum(u))
        self.t_hist.append(self.t)

        return self.x

    def plot(self,target=None):
        xx = np.array(self.x_hist)
        uu = np.array(self.u_hist)
        tt = np.array(self.t_hist)
        steps = xx.shape
        steps = xx.shape[0]
        target = target[:steps]

        p0, = plt.plot(tt,xx[:,0,0], label='pos')
        p1, = plt.plot(tt,xx[:,1,0], label='vel')
        p2, = plt.plot(tt,uu[:], label='ctrl')
        p3, = plt.plot(tt,target, label='target')
        p4, = plt.plot(tt,xx[:,0,0]-target, label='error')
        plt.legend(handles=[p0,p1,p2,p3,p4])
        plt.show()


if __name__ == "__main__":
    sim = Box()
    dt = 0.03
    T = 20
    n = 2
    p = 40
    du_max = 1
    u_max = 100


    mpc = MPC()
    A = sim.A
    B = sim.B
    tt = np.linspace(0,T,int(T/dt)+1)
    xref = np.vstack([(np.sin(tt)+1),np.zeros_like(tt)]).T
    xref = xref.reshape([-1,n,1])
    P = np.diag([1,0])
    Q = 0

    tic = time()
    for i in range(int(T/dt) + 1 - p):
        x0 = sim.x
        I2 = np.eye(2)
        mpc.convertLti(I2+A*dt,B*dt,P,Q,xref[i:i+p,:,:],x0,p,du_max,u_max)
        uu = np.array(mpc.solve())
        #print(uu)
        sim.step(uu[0,:],dt=dt)
    tac = time()

    print(1.0/(tac-tic)*(int(T/dt) - p))
    sim.plot(target=xref[:,0,0])


