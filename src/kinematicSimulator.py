# kinematic bicycle simulator
# for testing EKF on vicon
import random
import numpy as np
from time import time
from math import sin,cos,degrees,radians,tan
from kalmanFilter import KalmanFilter
import matplotlib.pyplot as plt

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
        self.observation_var = (5e-3)**2
        # error on action in prediction
        self.action_var = [radians(5)**2,0.5**2]
        random.Random(time())

        if (X is None):
            self.X = np.zeros([self.state_count,1])
            self.X = np.matrix(self.X)
        else:
            self.X = X

        self.true_state_vec = []
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
        x_obs = self.X[0,0] + random.gauss(0,0.005)
        y_obs = self.X[1,0] + random.gauss(0,0.005)
        theta_obs = self.X[3,0] + random.gauss(0,radians(2))
        return (x_obs,y_obs,theta_obs)
        # FIXME
        #return (self.X[0,0],self.X[1,0],self.X[3,0])

    def getNoisyAction(self,action=None):
        if action is None:
            alpha = 0.03
            self.action = np.array(self.action)*(1-alpha) + np.array([radians(random.uniform(-30,30)),random.uniform(-3,3)])*alpha
            # FIXME 
            self.action[0] = 0
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
    for i in range(steps):
        #action = (radians(10),3)
        noisy_action = sim.getNoisyAction()
        print(noisy_action)

        kf.predict(noisy_action,timestamp = dt*i)
        #print("predicted kf.X = "+str(kf.X))

        z = sim.getNoisyObservation()
        z = np.matrix(z).reshape(3,1)
        kf.update(z,timestamp = dt*i)
        #print("updated kf.X = "+str(kf.X))

        sim.true_state_vec.append(sim.X.copy())
        sim.kf_state_vec.append(kf.X.copy())
        sim.kf_K_vec.append(np.mean(kf.K))

        #print(kf.X[0,0]-sim.X[0,0])

        # go to next time step
        sim.step(dt)
        #print("sim.X = "+str(sim.X))

    # pos,x
    ax1 = plt.subplot(3,2,1)
    true_pos_x = [x[0,0] for x in sim.true_state_vec]
    kf_pos_x = [x[0,0] for x in sim.kf_state_vec]
    ax1.plot(true_pos_x)
    ax1.plot(kf_pos_x,linestyle='--')
    ax1.set_title("pos,x")
    # vel
    ax2 = plt.subplot(3,2,2)
    true_vel = [x[2,0] for x in sim.true_state_vec]
    kf_vel = [x[2,0] for x in sim.kf_state_vec]
    ax2.plot(true_vel)
    ax2.plot(kf_vel,linestyle='--')
    ax2.set_title("vel")
    # theta
    ax3 = plt.subplot(3,2,3)
    true_theta = [x[3,0] for x in sim.true_state_vec]
    kf_theta = [x[3,0] for x in sim.kf_state_vec]
    ax3.plot(true_theta)
    ax3.plot(kf_theta,linestyle='--')
    ax3.set_title("theta")
    # omega
    ax4 = plt.subplot(3,2,4)
    true_omega = [x[4,0] for x in sim.true_state_vec]
    kf_omega = [x[4,0] for x in sim.kf_state_vec]
    ax4.plot(true_omega)
    ax4.plot(kf_omega,linestyle='--')
    ax4.set_title("omega")
    # traj
    ax5 = plt.subplot(3,2,5)
    true_pos_y = [x[1,0] for x in sim.true_state_vec]
    kf_pos_y = [x[1,0] for x in sim.kf_state_vec]
    ax5.plot(true_pos_x,true_pos_y)
    ax5.plot(kf_pos_x,kf_pos_y,linestyle='--')
    ax5.set_title("trajectory")

    # Kalman gain K
    ax6 = plt.subplot(3,2,6)
    ax6.plot(sim.kf_K_vec,linestyle='--')
    ax6.set_title("kalman gain(1->observe)")

    plt.show()


if __name__ == "__main__":
    run(300)



