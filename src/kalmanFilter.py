# Kalman Filter for simple vehicle tracking using kinematic bicycle model
# requires python3
import numpy as np
from time import time
import random
from math import sin,cos,radians,degrees,tan
import matplotlib.pyplot as plt
import warnings


class KalmanFilter():
    def __init__(self,wheelbase):
        # wheelbase in meter
        self.wheelbase = wheelbase
        # maximum allowable steering angle
        self.max_steering = radians(35.0)

        # timestamp associated with current state
        self.state_ts = None
        # x,y coordinate(m), longitudinal velocity(m/s), vehicle heading(rad,ccw), angular speed(rad/s,ccw)
        # (x,y,v,theta,omega)
        self.state_count = 5
        self.action_count = 2
        self.X = None

        # TODO verify these...
        # variance of action
        self.action_var = [radians(1)**2,0.1**2]
        self.action_cov_mtx = np.diag(self.action_var)

        # var of noise in observation
        # from observation of a steady object:
        # bound of error, ~100 samples
        # x,y: 0.0002 m
        # theta: 0.5 deg
        self.var_xy = 0.005**2
        self.var_theta = radians(0.1)**2

        self.H = np.zeros([3,self.state_count])
        self.H[0,0] = 1
        self.H[1,1] = 1
        self.H[2,3] = 1

        self.R = np.diag([self.var_xy,self.var_xy,self.var_theta])
        self.R = np.matrix(self.R)
        self.last_steering = 0
        return

    # vehicle must be stationary
    def init(self,x=None,y=None,theta=None, timestamp=None):
        if timestamp is None:
            self.state_ts = time()
        else:
            self.state_ts = timestamp
        if x is None:
            # default know nothing initialization
            self.X = np.zeros([self.state_count,1])
            self.X = np.matrix(self.X)
            self.P = np.diag([100,100,0.1,radians(360),0.1])
            self.P = np.matrix(self.P)

        else:
            # perfect initialization
            self.X = np.zeros([self.state_count,1])
            self.X = np.matrix(self.X)
            self.X[0,0] = x
            self.X[1,0] = y
            self.X[3,0] = theta
            self.P = np.zeros([self.state_count,self.state_count])
            self.P = np.matrix(self.P)
            self.P[0,0] = 0.1
            self.P[1,1] = 0.1
            self.P[2,2] = 0.1
            self.P[3,3] = 0.1
            self.P[4,4] = 0.1

        return
        
    # action: steering angle(rad, left pos), longitudinal acceleration(m/s2)
    def predict(self,action,timestamp=None):
        if timestamp is None:
            timestamp = time()
        dt = timestamp - self.state_ts

        v = self.X[2,0]
        heading = self.X[3,0]
        omega = self.X[4,0]
        steering = action[0]
        acc_long = action[1]

        if (abs(steering)>self.max_steering):
            warnings.warn("Extreme steering value, %f"%steering)
            steering = np.clip(steering,-self.max_steering,self.max_steering)

        # update state matrix
        # x' = F(x,u) + B(u)(noise)
        self.F = np.zeros([self.state_count,self.state_count])

        # x' means x at next step
        self.F[0,0] = 1
        self.F[0,2] = cos(heading)*dt
        self.F[0,3] = -v*sin(heading)*dt
        self.F[1,1] = 1
        self.F[1,2] = sin(heading)*dt
        self.F[1,3] = v*cos(heading)*dt
        self.F[2,2] = 1
        self.F[3,3] = 1
        self.F[3,4] = dt
        self.F[4,2] = 1.0/self.wheelbase/cos(steering)**2*(steering-self.last_steering)
        self.F[4,4] = 1
        self.F = np.matrix(self.F)

        self.B = np.zeros([self.state_count,self.action_count])
        self.B[2,1] = dt
        # NOTE ignored small items
        self.B[4,0] = acc_long/self.wheelbase/cos(steering)**2*dt
        self.B[4,1] = dt/self.wheelbase*tan(steering)

        # predict
        self.X[0,0] += v*cos(heading)*dt
        self.X[1,0] += v*sin(heading)*dt
        self.X[2,0] += acc_long*dt
        self.X[3,0] += omega*dt
        self.X[4,0] += acc_long/self.wheelbase*tan(steering)*dt + v/self.wheelbase/cos(steering)**2*(steering-self.last_steering)

        self.last_steering = steering

        # update covariance matrix
        # TODO find a better Q
        # Maybe use ALS etc?
        #self.Q = np.diag([(0.05*dt)**2,(0.05*dt)**2,(0.05*dt)**2,(radians(10)*dt)**2,(radians(10)*dt)**2])*0
        self.Q = np.diag([0.0]*5)

        self.P = self.F @ self.P @ self.F.T + self.B @ self.action_cov_mtx @ self.B.T + self.Q
        self.P = self.F @ self.P @ self.F.T + self.Q

        self.state_ts = timestamp
        return self.X

    # update given z(observation) and associated timestamp
    # z should be a column vector consisting [x(m),y,theta(rad)].T
    def update(self,z,timestamp=None):
        y = z - self.H @ self.X
        S = self.H @ self.P @ self.H.T + self.R

        self.K = self.P @ self.H.T @ np.linalg.inv(S)

        self.X = self.X + self.K @ y
        self.P = (np.identity(self.state_count) - self.K @ self.H) @ self.P

        if timestamp is None:
            timestamp = time()
        self.state_ts = timestamp

        return

    def getState(self):
        return (self.X[0,0],self.X[1,0],self.X[2,0],self.X[3,0],self.X[4,0],self.X[5,0],self.X[6,0],self.X[7,0])


if __name__ == "__main__":
    # a simulator to test performance
    kf = KalmanFilter()
    x = 0
    vx = 0
    y = 0
    vy = 0
    theta = 0
    vtheta = 0

    x_real = []
    x_estimate = []
    vx_real = []
    vx_estimate = []
    vx_simple_hist = []
    err_percent = []
    err_base_percent = []
    zx_hist = [0]
    zy_hist = [0]

    step_size = 0.01
    random.seed()
    kf.init(timestamp=0)
    for i in range(1000):
        # 5m2/s stddev
        dir_a = random.random() * np.pi*2.0
        mag_a = random.random() * 5
        a_x = cos(dir_a)*mag_a
        a_y = sin(dir_a)*mag_a
        a_theta = (random.random()-0.5)*2*radians(2)
        theta += step_size*vtheta + 0.5*step_size**2*a_theta
        vtheta += step_size*a_theta
        
        x += step_size*vx + 0.5*step_size**2*a_x
        y += step_size*vy + 0.5*step_size**2*a_y
        vx += step_size*a_x
        vy += step_size*a_y

        z = np.matrix([[x,y,theta]]).T
        z[0,0] += 2*(random.random()-0.5)* 0.005
        zx_hist.append(z[0,0])
        vx_simple = z[0,0]-zx_hist[-2]
        vx_simple_hist.append(vx_simple)

        z[1,0] += 2*(random.random()-0.5)* 0.005
        zy_hist.append(z[1,0])
        vy_simple = z[1,0]-zy_hist[-2]

        z[2,0] += 2*(random.random()-0.5)* radians(2)

        kf.predict(timestamp=i*step_size)
        kf.update(z,timestamp=i*step_size)
        x_kf,vx_kf,ax_kf,y_kf,vy_kf,ay_kf,theta_kf,vtheta_kf = kf.getState()

        x_real.append(x)
        vx_real.append(vx)
        x_estimate.append(x_kf)
        vx_estimate.append(vx_kf)
        err_percent.append(((vx-vx_kf)**2+(vy-vy_kf)**2)**0.5/(vx**2+vy**2)**0.5*100)
        err_base_percent.append(((vx-vx_simple)**2+(vy-vy_simple)**2)**0.5/(vx**2+vy**2)**0.5*100)

        if i%10 == 0:
            #print(((x-x_kf)**2+(y-y_kf)**2)**0.5,(theta-theta_kf),kf.P[0,0])
            print(((vx-vx_kf)**2+(vy-vy_kf)**2)**0.5/(vx**2+vy**2)**0.5*100,(vtheta-vtheta_kf)/vtheta*100,kf.P[1,1])

    avg_err_percent_speed = np.mean(np.array(err_percent))
    avg_err_base_percent_speed = np.mean(np.array(err_base_percent))
    print("error in speed "+str(avg_err_percent_speed) +"%")
    print("error in speed (base)"+str(avg_err_base_percent_speed) +"%")
    print("kf is "+str(avg_err_base_percent_speed/avg_err_percent_speed) +" times better")
    plt.plot(x_real)
    plt.plot(x_estimate)
    plt.plot(zx_hist[1:])
    plt.legend(['X Real','X KF','X Direct'])
    plt.show()


    plt.plot(vx_real)
    plt.plot(vx_estimate)
    plt.plot(vx_simple_hist)
    plt.legend(['Vx Real','Vx KF','Vx Simple'])
    plt.show()


    

