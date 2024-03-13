# simple implementation of Unscented Kalman Filter for single parameter estimation
from common import *
from math import exp
import numpy as np

class UKF(PrintObject):

    def __init__(self,init_val=0):
        # current estimate for parameter
        self.mean = init_val
        # covariance for parameter
        self.cov = 1
        # weight for the mean
        self.W = 1/3

    def getSigmaPoints(self):
        #sigma = [self.mean, self.mean+self.cov, self.mean-self.cov]
        sigma = [self.mean]
        sigma.append(self.mean+(1/(1-self.W))**0.5*self.cov**0.5)
        sigma.append(self.mean-(1/(1-self.W))**0.5*self.cov**0.5)
        return sigma

    # zs: h(sigma_points) dim: n (measurement dim) * N(# of sigma points)
    # zx: h(ground truth) + noise dim: n*1
    def update(self,zs,zx):
        # no prediction needed, static dynamics
        # update
        weights = np.array([self.W, (1-self.W)/2, (1-self.W)/2])
        sigmas = np.array(self.getSigmaPoints()).reshape(-1,1)
        z_hat = zs @ weights.reshape((-1,1))
        R = np.diag([10.0]*4)

        S = np.sum([weights[i]* (zs[:,[i]]-z_hat) @ (zs[:,[i]]-z_hat).T for i in range(3)],axis=0) + R
        C = np.sum([weights[i]* (sigmas[i] - self.mean) * (zs[:,[i]]-z_hat).T for i in range(3)],axis=0)
        K = C @ np.linalg.inv(S)
        self.mean += (K @ (zx - z_hat)).item()
        self.cov +=  (-K @ S @ K.T).item()



