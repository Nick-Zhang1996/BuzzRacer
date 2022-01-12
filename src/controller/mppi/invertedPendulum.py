from math import sin
import numpy as np
import matplotlib.pyplot as plt
from mppi import MPPI
# inverted pendulum simulator

class InvertedPendulum:

    def __init__(self,x0=[0,0]):
        self.m = 1
        self.L = 1
        self.g = 9.81
        self.x = np.array(x0)
        self.t = 0
        self.x_hist = []
        self.t_hist = []
        self.u_hist = []

    def step(self,dt,u):
        g = self.g
        m = self.m
        L = self.L

        self.x[0] += self.x[1]*dt
        self.x[0] = (self.x[0] + np.pi) % (2*np.pi) - np.pi
        self.x[1] += (u - m*g*L*sin(self.x[0]))*dt
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
        ax.plot(tt,xx[:,0],label="angle")
        ax.plot(tt,xx[:,1],label="angular speed")
        ax.plot(tt,uu,label="u")
        ax.plot(tt,(xx[:,0])%(2*np.pi)-np.pi,label="error")
        ax.legend()
        plt.show()


if __name__=="__main__x":
    main = InvertedPendulum(x0=[0,0.1])

    dt = 0.01
    D = 2
    u_max = 0
    for i in range(int(10/dt)):
        u = 2*main.m*main.g*main.L*np.sin(main.x[0]) - D*main.x[1]
        if (u>u_max):
            u_max = u
        main.step(dt,u)
    main.plot()
    print(u_max)

def dynamics(state,control,dt):
    m = 1
    g = 9.81
    L = 1
    x = state
    u = control
    x[0] += x[1]*dt
    x[0] = (x[0] + np.pi) % (2*np.pi) - np.pi
    x[1] += (u[0] - m*g*L*sin(x[0]))*dt

    x[2] += x[3]*dt
    x[2] = (x[2] + np.pi) % (2*np.pi) - np.pi
    x[3] += (u[1] - m*g*L*sin(x[2]))*dt

    return x

def cost(state):
    x = state
    cost = ((x[2]-np.pi + np.pi)%(2*np.pi)-np.pi)**2 + 2.3*(x[3])**2
    cost += ((x[0]-np.pi + np.pi)%(2*np.pi)-np.pi)**2 + 0.1*(x[1])**2
    return cost


if __name__=="__main__":
    dt = 0.02

    main = InvertedPendulum(x0=[np.pi/2.0,0.0])

    noise = 10

    samples_count = 1000
    horizon_steps = 20
    control_dim = 1
    temperature = 1
    noise_cov = np.eye(control_dim)*noise*noise

    mppi = MPPI(samples_count,horizon_steps,control_dim,temperature,dt,noise_cov)
    # define dynamics
    #mppi.applyDiscreteDynamics = dynamics
    #mppi.evaluateCost = cost

    while (main.t<3):
        print(" sim t = %.2f"%(main.t))
        # state, ref_control, control limit
        uu = mppi.control_single(main.x,[0]*horizon_steps,[[-50,50]])
        main.step(dt,uu[0,0])

    mppi.p.summary()
    main.plot()



