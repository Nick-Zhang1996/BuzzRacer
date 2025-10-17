from time import time
from common import *
from track.RCPTrack import RCPTrack
class TofLocalization(Extension):
    def __init__(self, main):
        super().__init__(main)
        assert(isinstance(self.main.track,RCPTrack))
        # cars for which Tof localization is enabled
        self.car_ids = [0]

    def time(self):
        if (self.main.experiment_type == ExperimentType.Simulation):
            timestamp = self.main.sim_t
        else:
            timestamp = time()
        return timestamp

    def postInit(self):
        for car_id in self.car_ids:
            car = self.main.cars[car_id]
            car.tof_kf = KalmanFilter()
            # NOTE rely on car.states for initial localization
            car.tof_kf.init(car.states,self.time())
            car.tof_states = car.states

    # update state estimation given new tof measurements for a specific car
    def tofReadingUpdateCallback(self,car,timestamp=None):
        if (timestamp is None):
            timestamp = self.time()
        car.kf.update(car.tof_measurement,timestamp)

    # update car states
    def preUpdate(self):
        for car_id in self.car_ids:
            car = self.main.cars[car_id]
            t = time()
            if (self.main.experiment_type == ExperimentType.Simulation):
                t = self.main.sim_t
            action = (car.steering, car.throttle)
            (x,y,theta,v_forward,v_sideway,omega) = car.tof_kf.predict(action,timestamp=t)
            car.states = (x,y,theta,v_forward,v_sideway,omega)
        self.main.new_state_update.set()

    def update(self):
KalmanFilter(wheelbase=self.wheelbase)
                self.kf[-1].init(x_local,y_local,theta_local)
            self.kf[internal_id].predict(self.action)
            observation = np.matrix([[x_local,y_local,theta_local]]).T
            self.kf[internal_id].update(observation)


class KalmanFilter():
    def __init__(self):
        # timestamp associated with current state
        self.state_ts = None
        # unit: x,y coordinate(m), velocity(m/s), heading(rad,ccw), angular speed(rad/s,ccw)
        #(x,y,theta,v_forward,v_sideway,omega)
        self.state_dim = n = 6
        # (steering (rad, ccw+), throttle (-1,1))
        self.action_dim = m = 2
        # ToF [front, left, right, rear]
        self.measure_dim = h = 4

        # state, dim: (n,1)
        self.X = None
        # state covariance, dim: (n,n)
        self.P = None
        # dynamics noise, normalized by time
        self.q = np.diag([0.5]*n)
        self.action_cov_mtx = np.diag([0.1]*m)

        # Jacobian for dynamics, let dxdt = f(x,u)
        # f = df/dx, dim: (n,n)
        # f = None
        # Jacobian for discretized Jacobian F = f*dt
        # F = None
        # b = df/du, dim: (n,m)
        # b = None
        # Jacobian for discretized Jacobian B = b*dt
        # B = None
        # Jacobian for measurement function z = h(x), dim (h,n)
        # H = None
        # measurement error, dim (h,h)
        self.R = np.diag([0.1]*h)


        '''
        # measurement, dim:(h,1)

        # variance of action
        action_var = [radians(3)**2,1.5**2]
        # action noise
        self.R = np.matrix(self.R)
        # process noise
        self.Q = np.diag([0.005, 0.005, 6, 0.00005, 0.01])
        '''

    # initialize X (state) ts(time) P(covariance for state)
    def init(self,car_states,timestamp):
        self.X = np.array(car_states).reshape(self.state_dim,1)
        self.P = np.diag([0.1]*self.state_dim)
        self.state_ts = timestamp
        return

    # propagate dynamics with given action to [timestamp]
    def predict(self,action,timestamp):
        n = self.state_dim
        m = self.action_dim
        dt = (timestamp - self.state_ts)
        if (dt < 1e-10):
            return

        dxdt = Dynamics.f(self.X, action).reshape((n,1))

        dfdx, dfdu = Dynamics.df(self.X, action)
        F = np.eye(n) + dfdx.reshape(n,n) * dt
        B = dfdu.reshape(n,m) * dt
        Q = self.q * dt

        action = np.array(action).reshape((self.action_dim,1))

        self.X += dxdt * dt
        self.P = F @ self.P @ F.T + B @ self.action_cov_mtx @ B.T + Q
        self.state_ts = timestamp
        return

    # update given z(observation) and associated timestamp
    # NOTE this should be run right after prediction
    def update(self,z,timestamp):
        z = z.reshape((self.measure_dim,1))
        # TODO
        z_expected = make_measurement(self.X.flatten())
        H = measurement_jacobian()

        #y = z - H @ self.X
        y = z - z_expected
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.X += K @ y

        # wrap again for numerical stability
        self.X[2,0] = self.wrap(self.X[2,0])
        self.P = (np.identity(self.state_dim) - K @ H) @ self.P
        self.state_ts = timestamp

        return

class Dynamics:
    def __init__(self,car):
        self.n = 6
        self.m = 2
        self.car = car
        return

    def f(self, state, control):
        lf = self.car.lf
        lr = self.car.lr
        L = self.car.L

        Iz = self.car.Iz
        m = self.car.m

        # NOTE here vx = vf, vy = vs, different convention
        x,y,heading,vx,vy,omega = np.array(state).flatten()
        steering, throttle = np.array(control).flatten()

        # for small longitudinal velocity use kinematic model
        if (vx<0.05):
            beta = atan(lr/L*tan(steering))
            norm = lambda a,b:(a**2+b**2)**0.5
            # motor model
            d_vx = 6.17*(throttle - vx/15.2 -0.333)
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

            Ffy = tireCurve(slip_f) * m * 9.8 *lr/(lr+lf)
            Fry = 1.15*tireCurve(slip_r) * m * 9.8 *lf/(lr+lf)

            # Dynamics
            #d_vx = 1.0/m * (Frx - Ffy * np.sin( steering ) + m * vy * omega)
            d_vx = 6.17*(throttle - vx/15.2 -0.333)
            d_vy = 1.0/m * (Fry + Ffy * np.cos( steering ) - m * vx * omega)
            d_omega = 1.0/Iz * (Ffy * lf * np.cos( steering ) - Fry * lr)

            # discretization
            vx = vx + d_vx * dt
            vy = vy + d_vy * dt
            omega = omega + d_omega * dt 

        # back to global frame
        vxg = vx*cos(heading)-vy*sin(heading)
        vyg = vx*sin(heading)+vy*cos(heading)

        return np.array([vxg, vyg, omega, d_vx, d_vy, d_omega])

    # Jacobian of dynamics f(x,u), df/dx, df/du
    # TODO calculate this
    def df(self, x,u):
        dfdx = np.zeros((self.n,self.n))
        dfdu = np.zeros((self.n,self.m))
        return dfdx, dfdu
