# kinematic bicycle simulator
# for testing EKF on vicon
import random
import numpy as np
from time import time
from math import sin,cos,degrees,radians,tan
from kalmanFilter import KalmanFilter
import matplotlib.pyplot as plt
import progressbar

class kinematicSimulator():
    def __init__(self,X=None):
        # unit: SI unless noted
        # forward speed,m/s
        self.vf = 0
        # steering angle, rad, left pos
        self.steering = 0
        self.action = np.array([0,0])
        # longitudinal acceleration
        self.acc_l = 0
        self.state_count = 5
        self.wheelbase = 0.102

        # error on observation
        self.var_xy = 0.005**2
        self.var_theta = radians(0.1)**2
        # error on action in prediction
        self.action_var = [radians(0.1)**2,0.1**2]
        random.Random(time())

        if (X is None):
            self.X = np.zeros([self.state_count,1])
            self.X = np.matrix(self.X)
        else:
            self.X = X

        self.true_state_vec = []
        self.observed_vec = []
        self.kf_state_vec = []
        self.kf_K_vec = []

    # get random input to steering and longitudinal acceleration
    def randomInput(self):
        steering = radians(random.uniform(-30,30))
        acc = random.uniform(-3,3)
        return (steering,acc)

    def sinInput(self,ts):
        steering = radians(10)*sin(ts)
        acc = 3*sin(ts)
        return (steering,acc)

    def getNoisyObservation(self):
        x_obs = self.X[0,0] + random.gauss(0,self.var_xy**0.5)
        y_obs = self.X[1,0] + random.gauss(0,self.var_xy**0.5)
        theta_obs = self.X[3,0] + random.gauss(0,self.var_theta**0.5)
        return (x_obs,y_obs,theta_obs)

    def getNoisyAction(self,action=None):
        if action is None:
            alpha = 0.03
            self.action = np.array(self.action)*(1-alpha) + np.array([radians(10+random.uniform(-30,30)),random.uniform(-3,3)])*alpha
            action = self.action
        return (action[0]+random.gauss(0,self.action_var[0]**0.5),action[1]+random.gauss(0,self.action_var[1]**0.5))
        

    def step(self,dt,action=None):
        if action is None:
            action = self.action
        v = self.X[2,0]
        heading = self.X[3,0]
        steering = action[0]
        acc_long = action[1]
        # NOTE since updated state depends on previous state
        # the order we do this is important
        self.X[3,0] += self.X[4,0]*dt
        # NOTE this deviates from the report
        self.X[0,0] += v*cos(heading)*dt
        self.X[1,0] += v*sin(heading)*dt
        self.X[2,0] += acc_long*dt
        self.X[4,0] = self.X[2,0]/self.wheelbase * tan(steering)
        pass
        


def run(steps):
    dt = 0.01
    sim = kinematicSimulator()
    kf = KalmanFilter(sim.wheelbase)

    kf.init(timestamp=0.0)
    #for i in range(steps):
    for i in progressbar.progressbar(range(steps)):
        #action = (radians(10),3)
        noisy_action = sim.getNoisyAction()

        kf.predict(noisy_action,timestamp = dt*i)
        #print("predicted kf.X = "+str(kf.X))

        z = sim.getNoisyObservation()
        z = np.matrix(z).reshape(3,1)
        kf.update(z,timestamp = dt*i)
        #print("updated kf.X = "+str(kf.X))

        sim.observed_vec.append(z)
        sim.true_state_vec.append(sim.X.copy())
        sim.kf_state_vec.append(kf.X.copy())
        sim.kf_K_vec.append(np.mean(kf.K))

        #print(kf.X[0,0]-sim.X[0,0])

        # go to next time step
        sim.step(dt)
        #print("sim.X = "+str(sim.X))

    # pos,x
    tt = np.linspace(0,(steps-1)*dt,steps)
    ax1 = plt.subplot(3,2,1)
    true_pos_x = [x[0,0] for x in sim.true_state_vec]
    true_pos_y = [x[1,0] for x in sim.true_state_vec]
    kf_pos_x = [x[0,0] for x in sim.kf_state_vec]
    observed_x = [x[0,0] for x in sim.observed_vec]
    observed_y = [x[1,0] for x in sim.observed_vec]
    observed_theta = np.array([x[2,0] for x in sim.observed_vec])
    ax1.plot(tt,true_pos_x,label='Ground Truth')
    ax1.plot(tt,kf_pos_x,linestyle='--',label='EKF')
    ax1.plot(tt,observed_x,linestyle='-.',label='Observation')
    ax1.set_title("X Position (m)")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Position (m)")

    # vel
    ax2 = plt.subplot(3,2,2)
    true_vel = [x[2,0] for x in sim.true_state_vec]
    kf_vel = [x[2,0] for x in sim.kf_state_vec]
    d_x = np.array(observed_x)
    d_x = np.hstack([np.diff(d_x),0])
    d_y = np.array(observed_y)
    d_y = np.hstack([0,np.diff(d_y)])
    v_diff = (d_x**2+d_y**2)**0.5/dt

    ax2.plot(tt,np.abs(true_vel),label='Ground Truth')
    ax2.plot(tt,np.abs(kf_vel),linestyle='--',label='EKF')
    ax2.set_title("Longitudinal Velocity")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Velocity (m/s)")
    # theta
    ax3 = plt.subplot(3,2,3)
    true_theta = np.array([x[3,0] for x in sim.true_state_vec])
    kf_theta = np.array([x[3,0] for x in sim.kf_state_vec])
    ax3.plot(tt,true_theta*180.0/np.pi,label='Ground Truth')
    ax3.plot(tt,kf_theta*180.0/np.pi,linestyle='--',label='EKF')
    ax3.plot(tt,observed_theta*180.0/np.pi,linestyle='-.',label='Observation')
    ax3.set_title("Heading")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Heading (deg)")
    # omega
    ax4 = plt.subplot(3,2,4)
    true_omega = [x[4,0] for x in sim.true_state_vec]
    kf_omega = [x[4,0] for x in sim.kf_state_vec]
    ax4.plot(tt,true_omega,label='Ground Truth')
    ax4.plot(tt,kf_omega,linestyle='--',label='EKF')
    ax4.set_title("Angular Velocity")
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Angular Velocity (rad/s)")
    # traj
    ax5 = plt.subplot(3,2,5)
    true_pos_y = [x[1,0] for x in sim.true_state_vec]
    kf_pos_y = [x[1,0] for x in sim.kf_state_vec]
    ax5.plot(true_pos_x,true_pos_y,label='Ground Truth')
    ax5.plot(kf_pos_x,kf_pos_y,linestyle='--',label='EKF')
    ax5.set_title("Trajectory")
    ax5.set_xlabel("X Position (m)")
    ax5.set_ylabel("Y Position (m)")

    # Kalman gain K
    #ax6 = plt.subplot(3,2,6)
    #ax6.plot(tt,sim.kf_K_vec,linestyle='--')
    #ax6.set_title("kalman gain(1->observe)")

    plt.show()


if __name__ == "__main__":
    run(3000)



