import numpy as np
from math import sin,cos,tan,radians,degrees,pi
import matplotlib.pyplot as plt

# advanced dynamic simulator of mini z
class advCarSim:
    def __init__(self,x,y,heading):
        # front tire cornering stiffness
        g = 9.81
        self.m = 0.2
        self.Caf = 5*0.25*self.m*g
        self.Car = 5*0.25*self.m*g
        # longitudinal speed
        self.Vx = 1
        self.Vy = 0
        # CG to front axle
        self.lf = 0.05
        self.lr = 0.05
        # approximate as a solid box
        self.Iz = self.m/12.0*(0.1**2+0.1**2)

        self.x = x
        self.y = y
        self.psi = heading

        self.d_x = self.Vx*cos(self.psi)-self.Vy*sin(self.psi)
        self.d_y = self.Vx*sin(self.psi)+self.Vy*cos(self.psi)
        self.d_psi = 0
        self.states = np.array([self.x,self.d_x,self.y,self.d_y,self.psi,self.d_psi])
        self.states_hist = []
        self.local_states_hist = []
        self.norm = []
        # control signal: throttle(acc),steering
        self.throttle = 0
        self.steering = 0
        self.t = 0


    # update vehicle state
    # NOTE vx != 0
    # NOTE using car frame origined at CG with x pointing forward, y leftward
    def updateCar(self,dt,sim_states,throttle,steering):
        # simulator carries internal state and doesn't really need these
        '''
        x = sim_states['coord'][0]
        y = sim_states['coord'][1]
        psi = sim_states['heading']
        d_psi = sim_states['omega']
        Vx = sim_states['vf']
        '''

        self.t += dt
        # NOTE page 30 of book vehicle dynamics and control
        # ref frame vehicle CG, x forward y leftward
        # this is in car frame, rotate to world frame
        psi = self.states[4]
        # change ref frame to car frame
        self.Vx = self.states[1]*cos(psi) + self.states[3]*sin(psi)

        A = np.array([[0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, -(2*self.Caf+2*self.Car)/(self.m*self.Vx), 0, -self.Vx-(2*self.Caf*self.lf-2*self.Car*self.lr)/(self.m*self.Vx)],
                    [0, 0, 0, 0, 0, 1],
                    [0, 0, 0, -(2*self.lf*self.Caf-2*self.lr*self.Car)/(self.Iz*self.Vx), 0, -(2*self.lf**2*self.Caf+2*self.lr**2*self.Car)/(self.Iz*self.Vx)]])
        B = np.array([[0,1,0,0,0,0],[0,0,0,2*self.Caf/self.m,0,2*self.lf*self.Caf/self.Iz]]).T

        u = np.array([throttle,steering])
        # active roattion matrix of angle(rad)
        R = lambda angle: np.array([[cos(angle), 0,-sin(angle),0,0,0],
                        [0, cos(angle), 0,-sin(angle),0,0],
                        [sin(angle),0,cos(angle),0,0,0],
                        [0,sin(angle),0,cos(angle),0,0],
                        [0,0,0,0,1,0],
                        [0,0,0,0,0,1]])
        self.old_states = self.states.copy()
        self.states = self.states + R(psi) @ (A @ R(-psi) @ self.states + B @ u)*dt
        self.states_hist.append(self.states)
        self.local_states_hist.append(R(-psi)@self.states)

        self.throttle = throttle
        self.steering = steering

        coord = (self.states[0],self.states[2])
        heading = self.states[4]
        Vx = self.states[1]
        Vy = self.states[3]
        omega = self.states[5]
        sim_states = {'coord':coord,'heading':heading,'vf':Vx,'vs':Vy,'omega':omega}
        return sim_states
    def debug(self):
        data = np.array(self.states_hist)
        data_local = np.array(self.local_states_hist)

        print("x,y")
        plt.plot(data[:,0],data[:,2])
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

        print("dx")
        plt.plot(data_local[:,1])
        plt.show()
        print("dy")
        plt.plot(data_local[:,3])
        plt.show()

        print("psi")
        plt.plot(data[:,4]/pi*180.0)
        plt.show()
        print("omega")
        plt.plot(data[:,5])
        plt.show()


if __name__=='__main__':
    sim = advCarSim(0,0,radians(90))
    for i in range(200):
        throttle = 0
        steering = radians(10)
        sim.updateCar(0.005,None,throttle,steering)
    sim.debug()

        

