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

if __name__=="__main__":
    dt = 0.02

    main = InvertedPendulum(x0=[np.pi,0.1])
    horizon_steps = 20
    noise = 3
    mppi = MPPI(1000,horizon_steps,1,1,dt,noise)

    while (main.t<4):
        print(" sim t = %.2f"%(main.t))
        uu,_ = mppi.control_single(main.x,[0]*horizon_steps)
        #plt.plot(uu)
        #plt.show()
        #main.step(dt,uu[0])
        for u in uu:
            main.step(dt,u)

    main.plot()



