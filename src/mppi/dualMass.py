from math import sin
import numpy as np
import matplotlib.pyplot as plt
from mppi import MPPI
# dual mass simulator

class dualMass:

    def __init__(self,x0=[0,0,0,0]):
        self.m1 = 1
        self.m2 = 1
        self.k1 = 1
        self.k2 = 1
        self.c1 = 1.4
        self.c2 = 1.4

        self.x = np.array(x0,dtype=np.double)
        self.t = 0
        self.x_hist = []
        self.t_hist = []
        self.u_hist = []

    def step(self,dt,u):
        x1 = self.x[0]
        dx1 = self.x[1]
        x2 = self.x[2]
        dx2 = self.x[3]

        ddx1 = -(self.k1*x1 + self.c1*dx1 + self.k2*(x1-x2) + self.c2*(dx1-dx2)-u[0])/self.m1
        ddx2 = -(self.k2*(x2-x1) + self.c2*(dx2-dx1)-u[1])/self.m2

        x1 += dx1*dt
        dx1 += ddx1*dt
        x2 += dx2*dt
        dx2 += ddx2*dt

        self.x[0] = x1
        self.x[1] = dx1
        self.x[2] = x2
        self.x[3] = dx2

        self.x_hist.append(self.x.copy())
        self.t_hist.append(self.t)
        self.u_hist.append(u)
        self.t += dt

        return self.x

    def plot(self):
        xx = np.array(self.x_hist)
        tt = np.array(self.t_hist)
        uu = np.array(self.u_hist)

        fig = plt.figure()
        ax = fig.gca()
        ax.plot(tt,xx[:,0],label="x1")
        ax.plot(tt,xx[:,1],label="dx1")
        ax.plot(tt,xx[:,2],label="x2")
        ax.plot(tt,xx[:,3],label="dx2")
        ax.legend()
        plt.show()


if __name__=="__main__x":
    main = dualMass(x0=[0,0,0,0])
    # target: 1,0,3,0

    dt = 0.1
    for i in range(int(20/dt)):
        main.step(dt,[-1,2])
    main.plot()

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

def cost(state):
    R = np.diag([1,0.1,1,0.1])
    R = R**2
    x = np.array(state) - np.array([1,0,3,0])
    return x.T @ R @ x


if __name__=="__main__":
    dt = 0.1

    main = dualMass(x0=[0,0,0,0])

    noise = 2

    samples_count = 1000
    horizon_steps = 40
    control_dim = 2
    temperature = 1
    noise_cov = np.eye(control_dim)*noise*noise

    mppi = MPPI(samples_count,horizon_steps,control_dim,temperature,dt,noise_cov)
    # define dynamics
    mppi.applyDiscreteDynamics = dynamics
    mppi.evaluateCost = cost

    while (main.t<20):
        print(" sim t = %.2f"%(main.t))
        # state, ref_control, control limit
        uu = mppi.control_single(main.x,[[1,3]]*horizon_steps,[[-50,50]]*2)
        main.step(dt,uu[0,:])

    mppi.p.summary()
    main.plot()



