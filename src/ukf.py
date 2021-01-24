# Unscented kalman filter for joint-state (state+parameter) estimation
# on a dynamic bicycle model, following ETH's paper
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm,norm
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from math import radians

class UKF:
    def __init__(self,):
        self.state_n = 6
        self.param_n = 9

        # constants
        g = 9.81
        self.m = 0.1667
        self.lf = 0.09-0.036
        self.lr = 0.036

        # tire: Df,Dr,C,B
        # motor: cm1,cm2,cr, cd
        # car: Iz
        # TODO use a dictionary

        # initial values
        #Df, Dr, C, B, Cm1, Cm2, Cr, Cd = param
        # tire param
        Df = 1.0
        Dr = 1.0
        C = 1.4
        D = 1.0
        B = 0.714

        Cm1 = 7.0
        Cm2 = 0.0
        Cr = 0.24
        Cd = 0

        Iz = self.m/12.0*(0.1**2+0.1**2)
        
        self.param = np.array([Df, Dr, C, B, Cm1, Cm2, Cr, Cd, Iz])

        self.initSigmaPoints()
        
        # added to cov matrix at each step
        self.process_noise_cov = np.diag([0.01**2, 0.04**2, 0.01**2, 0.04**2, radians(2)**2, 3**2]+[1e-3**2]*self.param_n)
        self.observation_noise_cov = np.diag([0.02**2, 0.05**2, radians(2)**2])


    def initState(self,x,vxg,y,vyg,psi,omega):
        # state: x,vxg,y,vyg, psi, omega
        self.dynamic_state = np.array([0,0,0,0,0,0])
        self.dynamic_state[0] = x
        self.dynamic_state[1] = vxg
        self.dynamic_state[2] = y
        self.dynamic_state[3] = vyg
        self.dynamic_state[4] = psi
        self.dynamic_state[5] = omega

        self.state = np.hstack([self.dynamic_state,self.param])

        # cov matrix

        # 3 sigma
        self.state_3sigma = [0.1, 0.5, 0.1, 0.5,0.5, 10.0]
        self.param_3sigma = [1e-3**0.5]*self.param_n
        self.state_cov = (np.diag(self.state_3sigma + self.param_3sigma)/3.0)**2

    def initSigmaPoints(self):
        # scaling terms
        alfa = 1e-3
        beta = 0
        k = 0
        self.L = L = self.state_n + self.param_n
        # it's lambda, but lambda is taken, so its chinese ru
        ru = alfa*alfa*(L+k)-L
        self.gamma = gamma = (L + ru ) **0.5

        # weight
        self.w_m_0 = ru / (L + ru)
        self.w_c_0 = ru / (L + ru) + (1-alfa*alfa + beta)
        # w_m and w_c for index > 0
        self.w_i = 1.0/ (2* (L + ru))
        return

    # given mean and covariance matrix, generate sigma points
    def generateSigmaPoints(self,x,P):
        L = self.L

        # sigma points array
        x0 = np.array(x).reshape(-1,1)
        X = np.repeat(x0, 2*L+1, 1)

        def is_pos_def(x):
            return np.all(np.linalg.eigvals(x) > 0)
        '''
        if not is_pos_def(P):
            P = self.nearPD(P)
        '''
        variation = self.gamma * np.real(sqrtm(P))
        X[:,1:L+1] += variation
        X[:,L+1:] -= variation

        return X 

    #https://stackoverflow.com/questions/10939213/how-can-i-calculate-the-nearest-positive-semi-definite-matrix
    def nearPD(self,A, nit=10):
        def _getAplus(A):
            eigval, eigvec = np.linalg.eig(A)
            Q = np.matrix(eigvec)
            xdiag = np.matrix(np.diag(np.maximum(eigval, 0)))
            return Q*xdiag*Q.T

        def _getPs(A, W=None):
            W05 = np.matrix(W**.5)
            return  W05.I * _getAplus(W05 * A * W05) * W05.I

        def _getPu(A, W=None):
            Aret = np.array(A.copy())
            Aret[W > 0] = np.array(W)[W > 0]
        return np.matrix(Aret)
        n = A.shape[0]
        W = np.identity(n)
    # W is the matrix used for the norm (assumed to be Identity matrix here)
    # the algorithm should work for any diagonal W
        deltaS = 0
        Yk = A.copy()
        for k in range(nit):
            Rk = Yk - deltaS
            Xk = _getPs(Rk, W=W)
            deltaS = Xk - Rk
            Yk = _getPu(Xk, W=W)
        return Yk

    # progress var with covariance matrix P through fun(), unscented style
    # fun should have signature of:
    # fun(self,var,...), where var is np.array of size (var_length, batch_size)
    # fun must support batch processing 
    # return: mean, covariance matrix
    def unscentedTrans(self,var,P,fun, *args):
        sigma_x = self.generateSigmaPoints(var,P)
        post_sigma_x = fun(sigma_x, *args)
        # calc x_mean
        x_mean = np.sum(np.hstack([post_sigma_x[:,0].reshape(-1,1) * self.w_m_0, post_sigma_x[:,1:] * self.w_i]),axis=1)
        x_cov = self.w_c_0 * (post_sigma_x[:,0] - x_mean).reshape(-1,1) * (post_sigma_x[:,0] - x_mean)
        for i in range(1,2*self.L+1):
            x_cov = x_cov + self.w_i * (post_sigma_x[:,i] - x_mean).reshape(-1,1) * (post_sigma_x[:,i] - x_mean)

        return x_mean, x_cov

    # predict state using unscented transform
    # control dim: (control_dim)
    def predict(self, state, state_cov, control, dt):
        post_state, post_state_cov = self.unscentedTrans(state, state_cov, self.advanceModel, control, dt)

        # NOTE model noise
        post_state_cov = post_state_cov + self.process_noise_cov

        return post_state, post_state_cov


    # update from observation
    # y_real: actual measurement
    def update(self, state, state_cov, y_real):

        # predict measurement
        sigma_x = self.generateSigmaPoints(state, state_cov)
        x_mean = np.sum(np.hstack([sigma_x[:,0].reshape(-1,1) * self.w_m_0, sigma_x[:,1:] * self.w_i]),axis=1)

        sigma_y = sigma_x[(0,2,4),:]
        y_mean = np.sum(np.hstack([sigma_y[:,0].reshape(-1,1) * self.w_m_0, sigma_y[:,1:] * self.w_i]),axis=1)

        y_cov = self.w_c_0 * (sigma_y[:,0] - y_mean).reshape(-1,1) * (sigma_y[:,0] - y_mean)
        # NOTE observation noise
        y_cov = y_cov + self.observation_noise_cov
        for i in range(1,2*self.L+1):
            y_cov = y_cov + self.w_i * (sigma_y[:,i] - y_mean).reshape(-1,1) * (sigma_y[:,i] - y_mean)

        # calculate cross covariance mtx Pxy
        xy_cov = self.w_c_0 * (sigma_x[:,0] - x_mean).reshape(-1,1) * (sigma_y[:,0] - y_mean)
        for i in range(1,2*self.L+1):
            xy_cov = xy_cov + self.w_i * (sigma_x[:,i] - x_mean).reshape(-1,1) * (sigma_y[:,i] - y_mean)

        ukf_gain = xy_cov @ np.linalg.inv(y_cov)
        state = state + ukf_gain @ ( y_real - y_mean)
        state_cov = state_cov - ukf_gain @ y_cov @ ukf_gain.T

        return state, state_cov

    # give joint_state and control (throttle,steering)
    # advance model by dt
    def advanceModel(self,joint_state,control,dt):
        state = joint_state[:self.state_n,:]
        param = joint_state[self.state_n:,:]

        # note everybody's row vector
        x, vxg, y, vyg, psi, omega = state
        '''
        x = x.T
        vxg = vxg.T
        y = y.T
        vyg = vyg.T
        psi = psi.T
        omega = omega.T
        '''

        throttle, steering = control
        Df, Dr, C, B, Cm1, Cm2, Cr, Cd, Iz = param

        # convert to local frame
        vx = vxg * np.cos(psi) + vyg * np.sin(psi)
        vy = -vxg * np.sin(psi) + vyg * np.cos(psi)

        # tire model
        slip_f = - np.arctan( (omega * self.lf + vy)/vx ) + steering
        slip_r = np.arctan( (omega * self.lr - vy)/vx )

        # TODO add load transfer
        Ffy = Df * np.sin( C * np.arctan(B*slip_f)) * 9.8 * self.lr / (self.lr + self.lf) * self.m
        Fry = Dr * np.sin( C * np.arctan(B*slip_r)) * 9.8 * self.lf / (self.lr + self.lf) * self.m

        # motor model
        Frx = ( Cm1 - Cm2 * vx) * throttle - Cr - Cd * vx * vx

        # Dynamics
        d_vx = 1.0/self.m * (Frx - Ffy * np.sin( steering ) + self.m * vy * omega)
        d_vy = 1.0/self.m * (Fry - Ffy * np.cos( steering ) - self.m * vx * omega)
        d_omega = 1.0/Iz * (Ffy * self.lf * np.cos( steering ) - Fry * self.lr)

        # discretization
        vx = vx + d_vx * dt
        vy = vy + d_vy * dt
        omega = omega + d_omega * dt

        # convert back to global frame
        vxg = vx * np.cos(psi) - vy * np.sin(psi)
        vyg = vx * np.sin(psi) + vy * np.cos(psi)

        # update x,y,psi
        x = x + vxg * dt
        y = y + vyg * dt
        psi = psi + omega * dt

        new_state = (x, vxg, y, vyg, psi, omega)
        new_param = (Df, Dr, C, B, Cm1, Cm2, Cr, Cr, Iz)
        new_joint_state = np.vstack([new_state, new_param])

        return new_joint_state

# -------------- DEBUG --------------------
    def confidence_ellipse(self, mean,cov, ax, n_std=3.0, facecolor='none', **kwargs):
        """
        Create a plot of the covariance confidence ellipse of *x* and *y*.

        Parameters
        ----------

        ax : matplotlib.axes.Axes
            The axes object to draw the ellipse into.

        n_std : float
            The number of standard deviations to determine the ellipse's radiuses.

        **kwargs
            Forwarded to `~matplotlib.patches.Ellipse`

        Returns
        -------
        matplotlib.patches.Ellipse
        """
        pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
        # Using a special case to obtain the eigenvalues of this
        # two-dimensionl dataset.
        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)
        ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                          facecolor=facecolor, **kwargs)

        # Calculating the stdandard deviation of x from
        # the squareroot of the variance and multiplying
        # with the given number of standard deviations.
        scale_x = np.sqrt(cov[0, 0]) * n_std
        mean_x = mean[0]

        # calculating the stdandard deviation of y ...
        scale_y = np.sqrt(cov[1, 1]) * n_std
        mean_y = mean[1]

        transf = transforms.Affine2D() \
            .rotate_deg(45) \
            .scale(scale_x, scale_y) \
            .translate(mean_x, mean_y)

        ellipse.set_transform(transf + ax.transData)
        return ax.add_patch(ellipse)


    # test unscented covariance, using cartesian to polar coordinate transform
    # states: (x,y), dim:(state_dim, batch)
    # output: (rho, theta)
    def cartesianToPolar(self,states):
        theta = np.arctan2(states[1],states[0])
        rho = (states[0]**2 + states[1]**2)**0.5
        return np.vstack([rho,theta])

    def testCartesianToPolar(self):
        self.state_n = 2
        self.param_n = 0
        self.initSigmaPoints()
        # mean for test state
        # state: (x,y)
        state = np.array((5,9),dtype=np.float)
        # cov for test state
        state_P = np.identity(2)
        state_P[1,1] = 2

        #Monte Carlo method to get true output variance and mean
        mc_samples = np.random.multivariate_normal(state, state_P, size=(10000,)).T
        mc_outputs = self.cartesianToPolar(mc_samples)
        mc_mean = np.mean(mc_outputs,axis=1)
        mc_cov = np.cov(mc_outputs)

        print("mc mean")
        print(mc_mean)
        print("mc cov")
        print(mc_cov)

        #Unscented transform to simulate variance and mean
        u_mean, u_cov = self.unscentedTrans(state,state_P,self.cartesianToPolar)
        print("unscented mean")
        print(u_mean)
        print("unscented cov")
        print(u_cov)

        # plot
        fig = plt.figure()
        ax = fig.gca()
        #ax.plot(mc_samples, '*',label="mc samples")
        #ax.plot(mc_outputs, '*',label="mc ouputs")
        ax.plot(mc_mean[0],mc_mean[1], '+',label="mc mean")
        ax.plot(u_mean[0],u_mean[1], '+',label="unscented mean")

        self.confidence_ellipse(mc_mean, mc_cov, ax, edgecolor='red')
        self.confidence_ellipse(u_mean, u_cov, ax, edgecolor='blue')

        ax.legend()
        plt.show()


if __name__ =="__main__":
    ukf = UKF()
    #ukf.testCartesianToPolar()

