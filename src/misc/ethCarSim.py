# FIXME haven't adapted to extension yet
# using model in eth paper
import numpy as np
from math import sin,cos,tan,radians,degrees,pi,atan
import matplotlib.pyplot as plt
from tire import tireCurve

# advanced dynamic simulator of mini z
class ethCarSim:
    def __init__(self,x,y,heading,noise=False,noise_cov=None):

        # dimension
        self.lf = 0.09-0.036
        self.lr = 0.036
        self.L = 0.09
        # basic properties
        self.Iz = 0.00278
        self.m = 0.1667

        # tire model
        self.Df = 3.93731
        self.Dr = 6.23597
        self.C = 2.80646
        self.B = 0.51943

        # motor/longitudinal model
        self.Cm1 = 6.03154
        self.Cm2 = 0.96769
        self.Cr = -0.20375
        self.Cd = 0.00000

        # x,vx(global frame),y,vy,heading.omega(angular velocity)
        self.states = np.array([x,0,y,0,heading,0])

        self.state_dim = 6
        self.control_dim = 2
        
        self.noise = noise
        
        if noise:
            self.noise=True
            self.noise_cov = noise_cov
            assert np.array(noise_cov).shape == (6,6)
        else:
            self.noise=False

        self.states_hist = []
        self.local_states_hist = []
        self.norm = []
        # control signal: throttle(acc),steering
        self.throttle = 0
        self.steering = 0
        self.t = 0


    # update vehicle state
    # NOTE using car frame origined at CG with x pointing forward, y leftward
    # sim_states is no longer used but kept to maintain the same API
    # should be changed in next update
    def updateCar(self,dt,sim_states,throttle,steering):
        # simulator carries internal state and doesn't really need these
        lf = self.lf
        lr = self.lr
        L = self.L

        Df = self.Df
        Dr = self.Dr
        B = self.B
        C = self.C
        Cm1 = self.Cm1
        Cm2 = self.Cm2
        Cr = self.Cr
        Cd = self.Cd
        Iz = self.Iz
        m = self.m

        self.t += dt
        if (self.noise):
            process_noise_car = np.random.multivariate_normal([0.0]*self.state_dim, self.noise_cov, size=None, check_valid='warn', tol=1e-8)


        x = self.states[0]
        y = self.states[2]
        psi = heading = self.states[4]
        omega = self.states[5]
        # change ref frame to car frame

        # vehicle longitudinal velocity
        self.Vx = vx = self.states[1]*cos(psi) + self.states[3]*sin(psi)
        self.Vy = vy = -self.states[1]*sin(psi) + self.states[3]*cos(psi)

        # for small longitudinal velocity use kinematic model
        if (vx<0.05):
            beta = atan(lr/L*tan(steering))
            norm = lambda a,b:(a**2+b**2)**0.5
            # motor model
            d_vx = (( Cm1 - Cm2 * vx) * throttle - Cr - Cd * vx * vx)
            vx = vx + d_vx * dt
            vy = norm(vx,vy)*sin(beta)
            d_omega = 0.0
            omega = vx/L*tan(steering)

            slip_f = 0
            slip_r = 0
            Ffy = 0
            Fry = 0

        else:
            slip_f = -np.arctan((omega*lf + vy)/vx) + steering
            slip_r = np.arctan((omega*lr - vy)/vx)

            Ffy = Df * np.sin( C * np.arctan(B *slip_f)) * 9.8 * lr / (lr + lf) * m
            Fry = Dr * np.sin( C * np.arctan(B *slip_r)) * 9.8 * lf / (lr + lf) * m

            # motor model
            Frx = (( Cm1 - Cm2 * vx) * throttle - Cr - Cd * vx * vx)*m

            # Dynamics
            d_vx = 1.0/m * (Frx - Ffy * np.sin( steering ) + m * vy * omega)
            d_vy = 1.0/m * (Fry + Ffy * np.cos( steering ) - m * vx * omega)
            d_omega = 1.0/Iz * (Ffy * lf * np.cos( steering ) - Fry * lr)

            # discretization
            vx = vx + d_vx * dt
            vy = vy + d_vy * dt
            omega = omega + d_omega * dt 

            '''
            print("vx = %5.2f, vy = %5.2f"%(vx,vy))
            print("slip_f = %5.2f, slip_r = %5.2f"%(degrees(slip_f), degrees(slip_r)))
            print("f_coeff_f = %5.2f, f_coeff_f = %5.2f"%(tireCurve(slip_f), tireCurve(slip_r)))
            '''

        # add noise if need be
        if (self.noise):
            vx += process_noise_car[1] * dt
            vy += process_noise_car[3] * dt
            omega += process_noise_car[5] * dt

            x +=  process_noise_car[0] * dt
            y +=  process_noise_car[2] * dt
            heading += process_noise_car[4] * dt

        # back to global frame
        vxg = vx*cos(heading)-vy*sin(heading)
        vyg = vx*sin(heading)+vy*cos(heading)

        # update x,y, heading
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

# eth dynamic simulator
    def initEthSimulation(self,car,init_state = (0.3*0.6,1.7*0.6,radians(90))):
        car.new_state_update = Event()
        car.new_state_update.set()
        
        x,y,heading = init_state
        car.simulator = ethCarSim(x,y,heading,self.sim_noise,self.sim_noise_cov)
        # for keep track of time difference between simulation and reality
        # this allows a real-time simulation
        # here we only instantiate the variable, the actual value will be assigned in updateVisualization, since it takes quite a while to initialize the rest of the program
        self.real_sim_dt = None

        car.steering = steering = 0
        car.throttle = throttle = 0

        car.states = (x,y,heading,0,0,0)
        car.sim_states = {'coord':(x,y),'heading':heading,'vf':throttle,'vs':0,'omega':0}
        self.sim_dt = self.dt

    def updateEthSimulation(self,car):
        # update car
        sim_states = car.sim_states = car.simulator.updateCar(self.sim_dt,car.sim_states,car.throttle,car.steering)
        # (x,y,theta,vforward,vsideway=0,omega)
        car.states = np.array([sim_states['coord'][0],sim_states['coord'][1],sim_states['heading'],sim_states['vf'],sim_states['vs'],sim_states['omega']])
        if isnan(sim_states['heading']):
            print("error")
        #print(car.states)
        #print("v = %.2f"%(sim_states['vf']))
        car.new_state_update.set()

    def stopEthSimulation(self,car):
        return



if __name__=='__main__':
    sim = ethCarSim(0,0,radians(90))
    for i in range(400):
        throttle = 0.24 + 0.5
        steering = radians(10)
        print("step %d"%(i))
        sim.updateCar(0.01,None,throttle,steering)
        print(sim.states)
    sim.debug()

        

