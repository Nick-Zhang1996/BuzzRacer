# Kalman Filter for simple 2D object tracking
# requires python3
import numpy as np
from time import time
import random
from math import sin,cos,radians,degrees


class KalmanFilter():
    def __init__(self,):
        # timestamp associated with current state
        self.state_ts = None
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

        self.H = np.zeros([3,8])
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
            self.X = np.zeros([8,1])
            self.X = np.matrix(self.X)
            self.P = np.diag([100,0,0,100,0,0,100,0])
            self.P = np.matrix(self.P)

        else:
            # perfect initialization
            self.X = x
            self.P = np.zeros([8,8])
            self.P = np.matrix(self.P)

        return
        
    def predict(self,timestamp=None):
        if timestamp is None:
            timestamp = time()
        dt = timestamp - self.state_ts

        # update state matrix
        # x' = Fx (+ Ga)
        self.F = np.zeros([8,8])

        # x' means x at next step
        # x' = x + dt*dx + 0.5dt**2*ddx
        # dx' = dx + dt*ddx
        # ddx' = ddx
        self.F[0,0] = 1
        self.F[0,1] = dt
        self.F[0,2] = 0.5*dt*dt
        self.F[1,1] = 1
        self.F[1,2] = dt
        self.F[2,2] = 1

        # y' = y + dt*dy + 0.5dt**2*ddy
        # dy' = dy + dt*ddy
        # ddy' = ddy
        self.F[3,3] = 1
        self.F[3,4] = dt
        self.F[3,5] = 0.5*dt*dt
        self.F[4,4] = 1
        self.F[4,5] = dt
        self.F[5,5] = 1

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
        G = np.array([[0.5*dt*dt,dt,1,0.5*dt*dt,dt,1,dt,1]]).T
        self.Q = G @ G.T * self.var_acc
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
        self.P = (np.identity(8) - K @ self.H) @ self.P
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
        z[1,0] += 2*(random.random()-0.5)* 0.005
        z[2,0] += 2*(random.random()-0.5)* radians(2)

        kf.predict(timestamp=i*step_size)
        kf.update(z,timestamp=i*step_size)
        x_kf,vx_kf,ax_kf,y_kf,vy_kf,ay_kf,theta_kf,vtheta_kf = kf.getState()
        if i%10 == 0:
            print(((x-x_kf)**2+(y-y_kf)**2)**0.5,(theta-theta_kf),kf.P[0,0])


    

