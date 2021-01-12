# using model in eth paper
import numpy as np
from math import sin,cos,tan,radians,degrees,pi
import matplotlib.pyplot as plt
from tire import tireCurve

# advanced dynamic simulator of mini z
class ethCarSim:
    def __init__(self,x,y,heading,noise=False,noise_cov=None):
        # front tire cornering stiffness
        g = 9.81
        self.m = 0.1667
        # longitudinal speed
        self.Vx = 1.0
        self.Vy = 0
        # CG to front axle
        self.lf = 0.09-0.036
        self.lr = 0.036
        # approximate as a solid box
        self.Iz = 10*self.m/12.0*(0.1**2+0.1**2)

        self.x = x
        self.y = y
        self.psi = heading

        self.d_x = self.Vx*cos(self.psi)-self.Vy*sin(self.psi)
        self.d_y = self.Vx*sin(self.psi)+self.Vy*cos(self.psi)
        self.d_psi = 0
        self.states = np.array([self.x,self.d_x,self.y,self.d_y,self.psi,self.d_psi])

        self.state_dim = 6
        self.control_dim = 2
        
        self.noise = noise
        if noise:
            self.noise_cov = noise_cov
            assert np.array(noise_cov).shape == (6,6)

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


        x = self.states[0]
        y = self.states[2]
        psi = heading = self.states[4]
        omega = self.states[5]
        # change ref frame to car frame
        # vehicle longitudinal velocity
        self.Vx = vx = self.states[1]*cos(psi) + self.states[3]*sin(psi)
        self.Vy = vy = -self.states[1]*sin(psi) + self.states[3]*cos(psi)
        print("vx = %5.2f, vy = %5.2f"%(vx,vy))

        # TODO handle vx->0
        # for small velocity, use kinematic model 
        slip_f = -np.arctan((omega*self.lf + vy)/vx) + steering
        slip_r = np.arctan((omega*self.lr - vy)/vx)
        print("slip_f = %5.2f, slip_r = %5.2f"%(degrees(slip_f), degrees(slip_r)))

        # we call these acc but they are forces normalized by mass
        # TODO consider longitudinal load transfer
        lateral_acc_f = tireCurve(slip_f) * 9.8 * self.lr / (self.lr + self.lf)
        lateral_acc_r = tireCurve(slip_r) * 9.8 * self.lf / (self.lr + self.lf)

        # TODO use more comprehensive model
        forward_acc_r = (throttle - 0.24)*7.0

        print("acc_f = %5.2f, acc_r = %5.2f, forward = %5.2f"%(lateral_acc_f,lateral_acc_r,forward_acc_r))

        ax = forward_acc_r - lateral_acc_f * sin(steering) + vy*omega
        ay = lateral_acc_r + lateral_acc_f * cos(steering) - vx*omega

        vx += ax * dt
        vy += ay * dt

        # leading coeff = m/Iz
        d_omega = self.m/self.Iz*(lateral_acc_f * self.lf * cos(steering) - lateral_acc_r * self.lr )
        omega += d_omega * dt

        # back to global frame
        vxg = vx*cos(heading)-vy*sin(heading)
        vyg = vx*sin(heading)+vy*cos(heading)

        # apply updates
        # TODO add 1/2 a t2
        x += vxg*dt
        y += vyg*dt
        heading += omega*dt + 0.5* d_omega * dt * dt

        self.states = (x,vxg,y,vyg,heading,omega )


        self.states_hist.append(self.states)

        self.throttle = throttle
        self.steering = steering

        coord = (x,y)

        sim_states = {'coord':coord,'heading':heading,'vf':vx,'vs':vy,'omega':omega}
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
    sim = ethCarSim(0,0,radians(90))
    for i in range(400):
        throttle = 0.24 + 0.5
        steering = radians(10)
        print("step %d"%(i))
        sim.updateCar(0.01,None,throttle,steering)
        print(sim.states)
    sim.debug()

        

