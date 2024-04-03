from time import time
from math import sin,cos,tan,radians,degrees,atan
from common import *
from track.RCPTrack import RCPTrack
from extension.Extension import Extension
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
            car.tof_kf = KalmanFilter(Dynamics(car))
            car.tof_kf.tof_simulator = self.main.tof_simulator
            # NOTE rely on car.states for initial localization
            car.tof_kf.init(car.states,self.time())
            car.tof_states = car.states

    # update state estimation given new tof measurements for a specific car
    def tofReadingUpdateCallback(self,car,timestamp=None):
        if (timestamp is None):
            timestamp = self.time()
        car.tof_kf.update(car.tof_measurement,timestamp)

    # update car states
    def preUpdate(self):
        for car_id in self.car_ids:
            car = self.main.cars[car_id]
            t = self.time()
            action = (car.steering, car.throttle)
            (x,y,theta,v_forward,v_sideway,omega) = car.tof_kf.predict(action,timestamp=t)
            #car.states = (x,y,theta,v_forward,v_sideway,omega)
        self.main.new_state_update.set()

    def update(self):
        # update with measurement
        for car_id in self.car_ids:
            car = self.main.cars[car_id]
            self.tofReadingUpdateCallback(car)
        self.drawDebug()

    def drawDebug(self):
        # visualization NOTE very slow
        if (self.main.visualization.update_visualization.isSet()):
            img = self.main.visualization.visualization_img
            for car_id in self.car_ids:
                car = self.main.cars[car_id]
                px = car.tof_kf.P[0,0]
                py = car.tof_kf.P[1,1]
                p0 = car.tof_kf.X[:2,0]
                d = car.tof_kf.X[2,0]

                # draw state
                coord = self.main.track.m2canvas(p0)
                img = self.main.visualization.overlayCarRenderingRaw(img,car,coord,d)
                img = self.main.track.drawCircle(img, p0, 0.03)

                # draw covariance
                p1 = p0 + np.array([px*cos(d),px*sin(d)])
                img = self.main.track.drawPolyline([p0,p1],img)
                p2 = p0 + np.array([py*cos(d+np.pi/2),py*sin(d+np.pi/2)])
                img = self.main.track.drawPolyline([p0,p2],img)

            self.main.visualization.visualization_img = img
        return



class KalmanFilter():
    def __init__(self, dynamics):
        # timestamp associated with current state
        self.state_ts = None
        # unit: x,y coordinate(m), velocity(m/s), heading(rad,ccw), angular speed(rad/s,ccw)
        #(x,y,theta,v_forward,v_sideway,omega)
        self.state_dim = n = dynamics.n
        # (steering (rad, ccw+), throttle (-1,1))
        self.action_dim = m = dynamics.m
        # ToF [front, left, right, rear]
        self.measure_dim = h = 4

        # state, dim: (n,1)
        self.X = None
        # state covariance, dim: (n,n)
        self.P = None
        # dynamics noise, normalized by time
        # FIXME for more pronounced noise
        # self.q = np.diag([0.1,0.1,radians(5),0.5,0.5,radians(10)])*3
        self.q = np.diag([0.3,0.3,radians(5),0.5,0.5,radians(10)])*3
        self.action_cov_mtx = np.diag([0.1]*m)
        self.dynamics = dynamics

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
        self.R = np.diag([0.2]*h)



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
            return self.X.flatten()

        # FIXME artificial noise
        noise = (np.random.normal(size=n)*self.q.diagonal()).reshape((n,1))
        dxdt = self.dynamics.f(self.X, action).reshape((n,1)) + noise

        dfdx, dfdu = self.dynamics.df(self.X, action)
        F = np.eye(n) + dfdx.reshape(n,n) * dt
        B = dfdu.reshape(n,m) * dt
        Q = self.q * dt
        # print("dxdt:", dxdt)
        # print("F:", F)
        # print("B:", B)
        # print("Q:", Q)

        action = np.array(action).reshape((self.action_dim,1))

        self.X += dxdt * dt
        self.P = F @ self.P @ F.T + B @ self.action_cov_mtx @ B.T + Q
        self.state_ts = timestamp
        return self.X.flatten()

    def getTofRange(self,state):
        x,y,d,*_ = state
        # jac: drdx, drdy, drdd
        front,jac_front = self.tof_simulator.getTofReadingJacobian((x,y),d)
        left, jac_left  = self.tof_simulator.getTofReadingJacobian((x,y),d+np.pi/2)
        right,jac_right = self.tof_simulator.getTofReadingJacobian((x,y),d-np.pi/2)
        rear, jac_rear  = self.tof_simulator.getTofReadingJacobian((x,y),d+np.pi)
        '''
        # TODO verify jacobian
        # derivative against heading still isn't accurate, but doesn't affect perf
        
        # test front jacobian
        num_jac_front = []
        front_a ,_ = self.tof_simulator.getTofReadingJacobian((x+0.001,y),d)
        num_jac_front.append( (front_a-front)/0.001)
        front_a ,_ = self.tof_simulator.getTofReadingJacobian((x,y+0.001),d)
        num_jac_front.append( (front_a-front)/0.001)
        front_a ,_ = self.tof_simulator.getTofReadingJacobian((x,y),d+0.0001)
        num_jac_front.append( (front_a-front)/0.0001)

        num_jac_left = []
        left_a ,_ = self.tof_simulator.getTofReadingJacobian((x+0.001,y),d+np.pi/2)
        num_jac_left.append( (left_a-left)/0.001)
        left_a ,_ = self.tof_simulator.getTofReadingJacobian((x,y+0.001),d+np.pi/2)
        num_jac_left.append( (left_a-left)/0.001)
        left_a ,_ = self.tof_simulator.getTofReadingJacobian((x,y),d+np.pi/2+0.0001)
        num_jac_left.append( (left_a-left)/0.0001)

        err_front = np.array(jac_front)-np.array(num_jac_front)
        err_left = np.array(jac_left)-np.array(num_jac_left)


        if (np.max(np.abs(np.hstack([err_front,err_left]))) > 0.05):
            print(jac_front)
            print(num_jac_front)
            print(jac_left)
            print(num_jac_left)
            front,jac_front = self.tof_simulator.getTofReadingJacobian((x,y),d+0.0001)
        '''

        tof_range = np.array([front, left, right, rear]).reshape((self.measure_dim,1))
        jac = np.vstack([jac_front, jac_left, jac_right, jac_rear])
        jac = np.hstack([jac,np.zeros((4,3))])

        return (tof_range,jac)

    # update given z(observation) and associated timestamp
    # NOTE this should be run right after prediction
    def update(self,z,timestamp):
        z = np.array(z).reshape((self.measure_dim,1))
        z_expected,H = self.getTofRange(self.X.flatten())
        assert(z_expected.shape == (self.measure_dim,1))
        assert(H.shape == (self.measure_dim, self.state_dim))

        #y = z - H @ self.X
        y = z - z_expected
        # when a tof hit is near track edge, small uncertainty in state
        # can lead to drastically different expected measurement
        # this is due to the actual/expected tof hit edge is different
        # we remove these measurements
        # TODO requires refinement
        mask = (np.abs(y)<0.1).flatten()
        y = y[mask,:]
        H = H[mask,:]
        R = np.diag(np.diagonal(self.R)[mask])
        # print(H)

        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        # TODO  heading correction is finicky
        # limit updates
        # or do a particle filter on this
        correction = K @ y
        # limit update on heading to 1 deg
        correction[-1,:] = np.arctan(correction[-1,:]/(np.pi/2)*radians(2))/(np.pi/2)*radians(2)
        self.X += correction

        # wrap again for numerical stability
        self.X[2,0] = wrap(self.X[2,0])
        self.P = (np.identity(self.state_dim) - K @ H) @ self.P
        self.state_ts = timestamp

        return

class Dynamics:
    def __init__(self,car):
        self.n = 6
        self.m = 2
        self.car = car
        return

    def tireCurve(self,slip):
        C = 1.6
        B = 2.3
        D = 1.1
        # C: tail shape
        retval = D * np.sin( C * np.arctan(B *slip))
        return retval

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
            d_vy = 0
            d_omega = 0.0

        else:
            slip_f = -np.arctan((omega*lf + vy)/vx) + steering
            slip_r = np.arctan((omega*lr - vy)/vx)

            Ffy = self.tireCurve(slip_f) * m * 9.8 *lr/(lr+lf)
            Fry = 1.15*self.tireCurve(slip_r) * m * 9.8 *lf/(lr+lf)

            # Dynamics
            #d_vx = 1.0/m * (Frx - Ffy * np.sin( steering ) + m * vy * omega)
            d_vx = 6.17*(throttle - vx/15.2 -0.333)
            d_vy = 1.0/m * (Fry + Ffy * np.cos( steering ) - m * vx * omega)
            d_omega = 1.0/Iz * (Ffy * lf * np.cos( steering ) - Fry * lr)

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
