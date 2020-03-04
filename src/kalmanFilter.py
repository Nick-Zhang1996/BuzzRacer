# Kalman Filter for simple 2D object tracking
# requires python3
import numpy as np
from time import time
import random
from math import sin,cos,radians,degrees
import matplotlib.pyplot as plt


class KalmanFilter():
    def __init__(self,):
        # timestamp associated with current state
        self.state_ts = None
        self.state_count = 8
        # note: dx means x_dot, time derivative of x
        # (x,dx,ddx,y,dy,ddy,theta,dtheta)
        self.X = None

        # TODO verify these...
        # variance of linear acceleration in prediction
        # in meters
        self.var_acc = 5.0**2

        # var of noise in observation
        self.var_xy = 0.005**2
        self.var_theta = radians(2)**2

        self.H = np.zeros([3,self.state_count])
        self.H[0,0] = 1
        self.H[1,3] = 1
        self.H[2,6] = 1

        self.R = np.diag([self.var_xy,self.var_xy,self.var_theta])
        self.R = np.matrix(self.R)
        return

    def init(self,x=None,timestamp=None):
        if timestamp is None:
            self.state_ts = time()
        else:
            self.state_ts = timestamp
        if x is None:
            # default know nothing initialization
            self.X = np.zeros([self.state_count,1])
            self.X = np.matrix(self.X)
            self.P = np.diag([100,0,0,100,0,0,100,0])
            self.P = np.matrix(self.P)

        else:
            # perfect initialization
            self.X = x
            self.P = np.zeros([self.state_count,self.state_count])
            self.P = np.matrix(self.P)

        return
        
    def predict(self,timestamp=None):
        if timestamp is None:
            timestamp = time()
        dt = timestamp - self.state_ts

        # update state matrix
        # x' = Fx (+ Ga)
        self.F = np.zeros([self.state_count,self.state_count])

        # x' means x at next step
        # x' = x + dt*dx + 0.5dt**2*ddx
        # dx' = dx + dt*ddx
        # ddx' = ddx
        self.F[0,0] = 1
        self.F[0,1] = dt
        self.F[0,2] = 0.5*dt*dt
        self.F[1,1] = 1
        self.F[1,2] = dt
        # acc, here acc is disabled from the model
        self.F[2,2] = 0

        # y' = y + dt*dy + 0.5dt**2*ddy
        # dy' = dy + dt*ddy
        # ddy' = ddy
        self.F[3,3] = 1
        self.F[3,4] = dt
        self.F[3,5] = 0.5*dt*dt
        self.F[4,4] = 1
        self.F[4,5] = dt
        # acc disabled
        self.F[5,5] = 0

        # theta' = theta + dt*dtheta
        # dtheta' = dtheta
        self.F[6,6] = 1
        self.F[6,7] = dt
        self.F[7,7] = 1
        
        self.F = np.matrix(self.F)
        #print(self.X)
        self.X = self.F @ self.X
        #print(self.X)

        # update covariance matrix
        subG = np.array([[0.5*dt*dt,dt,1]]).T
        subQ = subG @ subG.T * self.var_acc
        self.Q = np.zeros([self.state_count,self.state_count])
        self.Q[0:3,0:3] = subQ
        self.Q[3:6,3:6] = subQ
        self.Q[6,6] = 0.25*dt**4*self.var_theta
        self.Q[6,7] = 0.5*dt**3*self.var_theta
        self.Q[7,6] = 0.5*dt**3*self.var_theta
        self.Q[7,7] = dt*dt*self.var_theta

        self.P = self.F @ self.P @ self.F.T + self.Q

        self.state_ts = timestamp
        return self.X

    # update given z(observation) and associated timestamp
    # z should be a column vector consisting [x(m),y,theta(rad)].T
    def update(self,z,timestamp=None):
        y = z - self.H @ self.X
        S = self.H @ self.P @ self.H.T + self.R

        K = self.P @ self.H.T @ np.linalg.inv(S)

        #print(self.X)
        #print(self.P[0,0])
        self.X = self.X + K @ y
        self.P = (np.identity(self.state_count) - K @ self.H) @ self.P
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


    

