from extension import Extension, ParticleFilter
from track import RCPTrack
from common import *
from time import time
from math import sin,cos,tan,radians,degrees,atan
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation



# For ToF (Time of Flight) localization through particle filter.

class TofLocalizationPf(Extension):
    
    def __init__(self, main):
        super().__init__(main)
        assert(isinstance(self.main.track, RCPTrack))
        # Cars for which Tof localization is enabled
        self.car_ids = [0]
        self.particle_count = 100

    def time(self):
        if (self.main.experiment_type == ExperimentType.Simulation):
            timestamp = self.main.sim_t
        else:
            timestamp = time()
        return timestamp

    def preInit(self):
        pass

    def init(self):
        pass

    def postInit(self):
        for car_id in self.car_ids:
            car = self.main.cars[car_id]
            car.tof_pf = ParticleFilter(Dynamics(car))
            car.tof_pf.tof_simulator = self.main.tof_simulator
            car.tof_pf.init(car.states, self.time(), self.particle_count)
            car.tof_states = car.states
            # self.visualizer = ParticleFilterVisualizer(car.tof_pf, self.main.track, car.states)

    def preUpdate(self):
        for car_id in self.car_ids:
            car = self.main.cars[car_id]
            t = self.time()
            action = (car.steering, car.throttle)
            car.tof_pf.predict(action,timestamp=t)
            # NOTE: coordinate definition: x = forward in car fixed body frame

        self.main.new_state_update.set()

    def update(self):
        for car_id in self.car_ids:
            car = self.main.cars[car_id]
            self.tofReadingUpdateCallback(car,self.main.track)

        self.main.new_state_update.set()
        # self.visualizer.update_plot()

    def tofReadingUpdateCallback(self, car, track, timestamp=None):
        if timestamp is None:
            timestamp = self.time()
            
        car.tof_pf.update(car.tof_measurement, self.main.track, timestamp)

    # def drawDebug(self):
    #     # visualization NOTE very slow
    #     if (self.main.visualization.update_visualization.isSet()):
    #         img = self.main.visualization.visualization_img
    #         for car_id in self.car_ids:
    #             car = self.main.cars[car_id]
    #             px = car.tof_kf.P[0,0]
    #             py = car.tof_kf.P[1,1]
    #             p0 = car.tof_kf.X[:2,0]
    #             d = car.tof_kf.X[2,0]

    #             # draw state
    #             coord = self.main.track.m2canvas(p0)
    #             img = self.main.visualization.overlayCarRenderingRaw(img,car,coord,d)
    #             img = self.main.track.drawCircle(img, p0, 0.03)

    #             # draw covariance
    #             p1 = p0 + np.array([px*cos(d),px*sin(d)])
    #             img = self.main.track.drawPolyline([p0,p1],img)
    #             p2 = p0 + np.array([py*cos(d+np.pi/2),py*sin(d+np.pi/2)])
    #             img = self.main.track.drawPolyline([p0,p2],img)

    #         self.main.visualization.visualization_img = img
    #     return

    def postUpdate(self):
        pass
    def preFinal(self):
        pass
    def final(self):
        pass
    def postFinal(self):
        pass


# class TofLocalizationPf(Extension):
    
#     def __init__(self, main):
#         super().__init__(main)
#         assert(isinstance(self.main.track, RCPTrack))
#         # Cars for which Tof localization is enabled
#         self.car_ids = [0]
#         self.particle_count = 100
        
#         # Set up real-time visualization
#         plt.ion()  # Turn on interactive mode
#         self.fig, self.ax = plt.subplots(figsize=(10, 8))

#     def time(self):
#         if (self.main.experiment_type == ExperimentType.Simulation):
#             timestamp = self.main.sim_t
#         else:
#             timestamp = time()
#         return timestamp

#     def preInit(self):
#         pass

#     def init(self):
#         pass

#     def postInit(self):
#         for car_id in self.car_ids:
#             car = self.main.cars[car_id]
#             car.tof_pf = RealTimeParticleFilter(Dynamics(car))
#             car.tof_pf.tof_simulator = self.main.tof_simulator
#             car.tof_pf.init(car.states, self.time(), self.particle_count)
#             car.tof_states = car.states

#     def preUpdate(self):
#         for car_id in self.car_ids:
#             car = self.main.cars[car_id]
#             t = self.time()
#             action = (car.steering, car.throttle)
#             car.tof_pf.predict(action, timestamp=t)

#     def update(self):
#         for car_id in self.car_ids:
#             car = self.main.cars[car_id]
#             self.tofReadingUpdateCallback(car, self.main.track)
#             self.main.new_state_update.set()

#     def tofReadingUpdateCallback(self, car, track, timestamp=None):
#         if timestamp is None:
#             timestamp = self.time()
#         car.tof_pf.update(car.tof_measurement, self.main.track, timestamp)

#     def postUpdate(self):
#         pass
#     def preFinal(self):
#         pass
#     def final(self):
#         pass
#     def postFinal(self):
#         pass


class ParticleFilter():
    def __init__(self, dynamics):
        self.state_ts = None
        # State space: (x, y, theta, v_x, v_y, omega)
        # note that x is forward in car fixed body frame
        self.state_dim = n = dynamics.n
        # (steering (rad, ccw+), throttle (-1, 1))
        self.action_dim = m = dynamics.m
        # ToF [front, left, right, rear]
        self.measure_dim = h = 4

        # State, dim: (n, 1)
        self.X = None
        # State covariance, dim: (n, n)
        self.P = None
        self.var = 0.00025
        # Dynamics noise, normalized by time
        self.q = np.diag([0.5, 0.5, radians(10), 0.5, 0.5, radians(10)])/1000
        # self.q = np.diag([0.5, 0.5, radians(5), 0.25, 0.25, radians(5)])/10

        self.action_cov_mtx = np.diag([0.1] * m)/1000
        self.dynamics = dynamics

        # Measurement error, dim: (h, h)
        self.R = np.diag([0.05] * h)
        
        # offset from car cg to tof sensor in each direction
        self.car_x_offset = 0.04
        self.car_y_offset = 0.01

    def init(self, car_state, timestamp, particle_count):
        self.X = np.array(car_state).reshape(self.state_dim,1)
        self.P = np.diag([self.var] * self.state_dim) #TODO: Verify covariance matrix for pf
        self.state_ts = timestamp
        self.particle_array = np.random.multivariate_normal(
            mean=self.X.flatten(),
            cov=self.P,
            size=particle_count
        )
        self.weights = np.zeros(particle_count)

    def predict(self,action,timestamp):
        self.advanceParticles(action,timestamp)
        self.state_ts = timestamp
    
    def update(self,measurements,track,timestamp):
        self.updateWeights(measurements,track)
        self.resampleParticles()

        self.state_ts = timestamp
        # self.plotParticles(track, self.X)
        self.X = np.average(self.particle_array,axis=0,weights=self.weights)

        # plt.scatter(self.particle_array[:, 0], self.particle_array[:, 1], c=self.weights, cmap='viridis')
        # plt.show()

    def plotParticles(self, track, car_state):
        plt.figure(figsize=(8, 8))
        plt.axis("equal")

        # Plot the track boundary
        # track.plotBoundary()

        # Plot the true car state
        plt.scatter(car_state[0], car_state[1], c='red', s=100, label='True State')

        # Plot the particles
        weights_normalized = self.weights / np.max(self.weights)
        plt.scatter(
            self.particle_array[:, 0],  # x-coordinates
            self.particle_array[:, 1],  # y-coordinates
            c=weights_normalized,       # color by weight
            cmap='viridis',
            s=20,                       # particle size
            label='Particles'
        )

        plt.colorbar(label='Normalized Weight')
        plt.legend()
        plt.title('Particle Filter Visualization')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.grid(True)
        plt.show()
    
    def advanceParticles(self,action,timestamp):
        # Advance particles with noisy dynamics
        n = self.state_dim
        m = self.action_dim
        dt = (timestamp - self.state_ts)
        if (dt < 1e-10):
            pass # TODO: Better handling for too small time step edge case

        noisy_actions = action + np.random.multivariate_normal(
            mean=np.zeros(self.action_dim),
            cov=self.action_cov_mtx,
            size=len(self.particle_array)
        )

        dynamics_noise = np.random.multivariate_normal(
            mean=np.zeros(self.state_dim),
            cov=self.q,
            size=len(self.particle_array)
        )

        dxdt = np.array([
            self.dynamics.f(particle.reshape(self.state_dim, 1), noisy_actions[i], dt).flatten()
            # self.dynamics.f(particle.reshape(self.state_dim, 1), noisy_actions[i]).flatten()
            for i, particle in enumerate(self.particle_array)
        ])

        self.particle_array += dxdt * dt + dynamics_noise
        self.state_ts = timestamp


    def updateWeights(self,measurements,track):
        R_inv = np.linalg.inv(self.R)
        normalizing_factor = np.sqrt((np.pi ** 2) ** self.measure_dim * np.linalg.det(self.R))
        
        predicted_measurements = np.array([
            self.getTofRange(self.particle_array[i]).flatten()
            for i in range(len(self.particle_array))
        ])

        residuals = measurements - predicted_measurements
        self.weights = np.exp(-0.5 * np.sum(residuals @ R_inv * residuals, axis=1)) / normalizing_factor
        # if self.weights > 0:
        if True:
            self.weights /= np.sum(self.weights)
        # TODO: numerical instability / degenerate case?
        # else:
        #     print('degenerate!')
        #     self.weights = np.ones(len(self.weights)) / len(self.weights)

    def getTofRange(self,state):
        x,y,d,*_ = state

        front = self.tof_simulator.getTofReading((x,y), d)
        left = self.tof_simulator.getTofReading((x,y), d + np.pi/2)
        right = self.tof_simulator.getTofReading((x,y), d - np.pi/2)
        rear = self.tof_simulator.getTofReading((x,y), d + np.pi)

        tof_range = np.array([front, left, right, rear]).reshape((self.measure_dim,1))
        # print(tof_range.flatten())
        return tof_range
    
    def resampleParticles(self):
        ess = 1.0 / np.sum(self.weights ** 2)

        ess_threshold = 0.40 * len(self.particle_array)

        if ess < ess_threshold:
        # if True:
            mask = np.random.choice(
                range(len(self.particle_array)),
                size = len(self.particle_array),
                p = self.weights
            )

            self.particle_array = self.particle_array[mask]
            self.weights = np.ones(len(self.particle_array)) / len(self.particle_array)

class RealTimeParticleFilter(ParticleFilter):
    def __init__(self, dynamics):
        super().__init__(dynamics)
        plt.ion()  # Turn on interactive mode
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.scatter = None
        self.true_state_plot = None
        
    def update(self, measurements, track, timestamp):
        # Perform standard particle filter update
        super().update(measurements, track, timestamp)
        
        # Real-time visualization
        self.ax.clear()
        self.ax.set_title('Particle Filter Visualization')
        
        # Plot particles
        weights_normalized = self.weights / np.max(self.weights)
        self.scatter = self.ax.scatter(
            self.particle_array[:, 0], 
            self.particle_array[:, 1], 
            c=weights_normalized, 
            cmap='viridis', 
            s=20, 
            alpha=0.7
        )
        
        # Plot true state
        self.true_state_plot = self.ax.scatter(
            self.X[0], 
            self.X[1], 
            c='red', 
            s=100, 
            marker='x', 
            label='True State'
        )
        
        self.ax.set_xlabel('X Position')
        self.ax.set_ylabel('Y Position')
        self.ax.legend()
        self.ax.grid(True)
        
        # Refresh the plot
        plt.pause(0.01)


class Dynamics():
    # ported from past tof projects: maybe needs verification
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

    def f(self, state, control, dt):
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

        return np.array([vxg, vyg, omega+0.5*d_omega*dt, d_vx, d_vy, d_omega])

    def fmatrix(self, state, control, dt):
        x,y,heading,vx,vy,omega = np.array(state).flatten()
        steering, throttle = np.array(control).flatten()

        x2 = dt*(-(dt*(6.17*throttle - 0.405921052631579*vx - 2.05461) + vx)*sin(heading) - (dt*(-1.0*omega*vx + 4.5472*sin(1.6*atan(2.3*steering - 2.3*atan((0.04824*omega + vy)/vx)))*cos(steering) + 6.04072*sin(1.6*atan(2.3*atan((0.04176*omega - vy)/vx)))) + vy)*cos(heading))
        x3 = dt*(-dt*(-1.0*omega + 16.733696*(0.04824*omega + vy)*cos(steering)*cos(1.6*atan(2.3*steering - 2.3*atan((0.04824*omega + vy)/vx)))/(vx**2*(1 + (0.04824*omega + vy)**2/vx**2)*(5.29*(steering - atan((0.04824*omega + vy)/vx))**2 + 1)) - 22.2298496*(0.04176*omega - vy)*cos(1.6*atan(2.3*atan((0.04176*omega - vy)/vx)))/(vx**2*(1 + (0.04176*omega - vy)**2/vx**2)*(5.29*atan((0.04176*omega - vy)/vx)**2 + 1)))*sin(heading) + (1 - 0.405921052631579*dt)*cos(heading))
        x4 = -dt*(dt*(-16.733696*cos(steering)*cos(1.6*atan(2.3*steering - 2.3*atan((0.04824*omega + vy)/vx)))/(vx*(1 + (0.04824*omega + vy)**2/vx**2)*(5.29*(steering - atan((0.04824*omega + vy)/vx))**2 + 1)) - 22.2298496*cos(1.6*atan(2.3*atan((0.04176*omega - vy)/vx)))/(vx*(1 + (0.04176*omega - vy)**2/vx**2)*(5.29*atan((0.04176*omega - vy)/vx)**2 + 1))) + 1)*sin(heading)
        x5 = -dt**2*(-1.0*vx - 0.80723349504*cos(steering)*cos(1.6*atan(2.3*steering - 2.3*atan((0.04824*omega + vy)/vx)))/(vx*(1 + (0.04824*omega + vy)**2/vx**2)*(5.29*(steering - atan((0.04824*omega + vy)/vx))**2 + 1)) + 0.928318519296*cos(1.6*atan(2.3*atan((0.04176*omega - vy)/vx)))/(vx*(1 + (0.04176*omega - vy)**2/vx**2)*(5.29*atan((0.04176*omega - vy)/vx)**2 + 1)))*sin(heading)

        y2 = dt*((dt*(6.17*throttle - 0.405921052631579*vx - 2.05461) + vx)*cos(heading) - (dt*(-1.0*omega*vx + 4.5472*sin(1.6*atan(2.3*steering - 2.3*atan((0.04824*omega + vy)/vx)))*cos(steering) + 6.04072*sin(1.6*atan(2.3*atan((0.04176*omega - vy)/vx)))) + vy)*sin(heading))
        y3 = dt*(dt*(-1.0*omega + 16.733696*(0.04824*omega + vy)*cos(steering)*cos(1.6*atan(2.3*steering - 2.3*atan((0.04824*omega + vy)/vx)))/(vx**2*(1 + (0.04824*omega + vy)**2/vx**2)*(5.29*(steering - atan((0.04824*omega + vy)/vx))**2 + 1)) - 22.2298496*(0.04176*omega - vy)*cos(1.6*atan(2.3*atan((0.04176*omega - vy)/vx)))/(vx**2*(1 + (0.04176*omega - vy)**2/vx**2)*(5.29*atan((0.04176*omega - vy)/vx)**2 + 1)))*cos(heading) + (1 - 0.405921052631579*dt)*sin(heading))
        y4 = dt*(dt*(-16.733696*cos(steering)*cos(1.6*atan(2.3*steering - 2.3*atan((0.04824*omega + vy)/vx)))/(vx*(1 + (0.04824*omega + vy)**2/vx**2)*(5.29*(steering - atan((0.04824*omega + vy)/vx))**2 + 1)) - 22.2298496*cos(1.6*atan(2.3*atan((0.04176*omega - vy)/vx)))/(vx*(1 + (0.04176*omega - vy)**2/vx**2)*(5.29*atan((0.04176*omega - vy)/vx)**2 + 1))) + 1)*cos(heading)
        y5 = dt**2*(-1.0*vx - 0.80723349504*cos(steering)*cos(1.6*atan(2.3*steering - 2.3*atan((0.04824*omega + vy)/vx)))/(vx*(1 + (0.04824*omega + vy)**2/vx**2)*(5.29*(steering - atan((0.04824*omega + vy)/vx))**2 + 1)) + 0.928318519296*cos(1.6*atan(2.3*atan((0.04176*omega - vy)/vx)))/(vx*(1 + (0.04176*omega - vy)**2/vx**2)*(5.29*atan((0.04176*omega - vy)/vx)**2 + 1)))*cos(heading)

        heading3 = dt**2*(161.057532995459*(0.04824*omega + vy)*cos(steering)*cos(1.6*atan(2.3*steering - 2.3*atan((0.04824*omega + vy)/vx)))/(vx**2*(1 + (0.04824*omega + vy)**2/vx**2)*(5.29*(steering - atan((0.04824*omega + vy)/vx))**2 + 1)) + 185.216162944778*(0.04176*omega - vy)*cos(1.6*atan(2.3*atan((0.04176*omega - vy)/vx)))/(vx**2*(1 + (0.04176*omega - vy)**2/vx**2)*(5.29*atan((0.04176*omega - vy)/vx)**2 + 1))) + dt**2*(322.115065990918*(0.04824*omega + vy)*cos(steering)*cos(1.6*atan(2.3*steering - 2.3*atan((0.04824*omega + vy)/vx)))/(vx**2*(1 + (0.04824*omega + vy)**2/vx**2)*(5.29*(steering - atan((0.04824*omega + vy)/vx))**2 + 1)) + 370.432325889556*(0.04176*omega - vy)*cos(1.6*atan(2.3*atan((0.04176*omega - vy)/vx)))/(vx**2*(1 + (0.04176*omega - vy)**2/vx**2)*(5.29*atan((0.04176*omega - vy)/vx)**2 + 1)))
        heading4 = dt**2*(-322.115065990918*cos(steering)*cos(1.6*atan(2.3*steering - 2.3*atan((0.04824*omega + vy)/vx)))/(vx*(1 + (0.04824*omega + vy)**2/vx**2)*(5.29*(steering - atan((0.04824*omega + vy)/vx))**2 + 1)) + 370.432325889556*cos(1.6*atan(2.3*atan((0.04176*omega - vy)/vx)))/(vx*(1 + (0.04176*omega - vy)**2/vx**2)*(5.29*atan((0.04176*omega - vy)/vx)**2 + 1))) + dt**2*(-161.057532995459*cos(steering)*cos(1.6*atan(2.3*steering - 2.3*atan((0.04824*omega + vy)/vx)))/(vx*(1 + (0.04824*omega + vy)**2/vx**2)*(5.29*(steering - atan((0.04824*omega + vy)/vx))**2 + 1)) + 185.216162944778*cos(1.6*atan(2.3*atan((0.04176*omega - vy)/vx)))/(vx*(1 + (0.04176*omega - vy)**2/vx**2)*(5.29*atan((0.04176*omega - vy)/vx)**2 + 1)))
        heading5 = dt**2*(-7.76941539170095*cos(steering)*cos(1.6*atan(2.3*steering - 2.3*atan((0.04824*omega + vy)/vx)))/(vx*(1 + (0.04824*omega + vy)**2/vx**2)*(5.29*(steering - atan((0.04824*omega + vy)/vx))**2 + 1)) - 7.73462696457393*cos(1.6*atan(2.3*atan((0.04176*omega - vy)/vx)))/(vx*(1 + (0.04176*omega - vy)**2/vx**2)*(5.29*atan((0.04176*omega - vy)/vx)**2 + 1))) + dt*(dt*(-15.5388307834019*cos(steering)*cos(1.6*atan(2.3*steering - 2.3*atan((0.04824*omega + vy)/vx)))/(vx*(1 + (0.04824*omega + vy)**2/vx**2)*(5.29*(steering - atan((0.04824*omega + vy)/vx))**2 + 1)) - 15.4692539291479*cos(1.6*atan(2.3*atan((0.04176*omega - vy)/vx)))/(vx*(1 + (0.04176*omega - vy)**2/vx**2)*(5.29*atan((0.04176*omega - vy)/vx)**2 + 1))) + 1)

        vy3 = dt*(-1.0*omega + 16.733696*(0.04824*omega + vy)*cos(steering)*cos(1.6*atan(2.3*steering - 2.3*atan((0.04824*omega + vy)/vx)))/(vx**2*(1 + (0.04824*omega + vy)**2/vx**2)*(5.29*(steering - atan((0.04824*omega + vy)/vx))**2 + 1)) - 22.2298496*(0.04176*omega - vy)*cos(1.6*atan(2.3*atan((0.04176*omega - vy)/vx)))/(vx**2*(1 + (0.04176*omega - vy)**2/vx**2)*(5.29*atan((0.04176*omega - vy)/vx)**2 + 1)))
        vy4 = dt*(-16.733696*cos(steering)*cos(1.6*atan(2.3*steering - 2.3*atan((0.04824*omega + vy)/vx)))/(vx*(1 + (0.04824*omega + vy)**2/vx**2)*(5.29*(steering - atan((0.04824*omega + vy)/vx))**2 + 1)) - 22.2298496*cos(1.6*atan(2.3*atan((0.04176*omega - vy)/vx)))/(vx*(1 + (0.04176*omega - vy)**2/vx**2)*(5.29*atan((0.04176*omega - vy)/vx)**2 + 1))) + 1
        vy5 = dt*(-1.0*vx - 0.80723349504*cos(steering)*cos(1.6*atan(2.3*steering - 2.3*atan((0.04824*omega + vy)/vx)))/(vx*(1 + (0.04824*omega + vy)**2/vx**2)*(5.29*(steering - atan((0.04824*omega + vy)/vx))**2 + 1)) + 0.928318519296*cos(1.6*atan(2.3*atan((0.04176*omega - vy)/vx)))/(vx*(1 + (0.04176*omega - vy)**2/vx**2)*(5.29*atan((0.04176*omega - vy)/vx)**2 + 1)))

        omega3 = dt*(322.115065990918*(0.04824*omega + vy)*cos(steering)*cos(1.6*atan(2.3*steering - 2.3*atan((0.04824*omega + vy)/vx)))/(vx**2*(1 + (0.04824*omega + vy)**2/vx**2)*(5.29*(steering - atan((0.04824*omega + vy)/vx))**2 + 1)) + 370.432325889556*(0.04176*omega - vy)*cos(1.6*atan(2.3*atan((0.04176*omega - vy)/vx)))/(vx**2*(1 + (0.04176*omega - vy)**2/vx**2)*(5.29*atan((0.04176*omega - vy)/vx)**2 + 1)))
        omega4 = dt*(-322.115065990918*cos(steering)*cos(1.6*atan(2.3*steering - 2.3*atan((0.04824*omega + vy)/vx)))/(vx*(1 + (0.04824*omega + vy)**2/vx**2)*(5.29*(steering - atan((0.04824*omega + vy)/vx))**2 + 1)) + 370.432325889556*cos(1.6*atan(2.3*atan((0.04176*omega - vy)/vx)))/(vx*(1 + (0.04176*omega - vy)**2/vx**2)*(5.29*atan((0.04176*omega - vy)/vx)**2 + 1)))
        omega5 = dt*(-15.5388307834019*cos(steering)*cos(1.6*atan(2.3*steering - 2.3*atan((0.04824*omega + vy)/vx)))/(vx*(1 + (0.04824*omega + vy)**2/vx**2)*(5.29*(steering - atan((0.04824*omega + vy)/vx))**2 + 1)) - 15.4692539291479*cos(1.6*atan(2.3*atan((0.04176*omega - vy)/vx)))/(vx*(1 + (0.04176*omega - vy)**2/vx**2)*(5.29*atan((0.04176*omega - vy)/vx)**2 + 1))) + 1
        
        F = np.array([[1.0, 0.0, x2, x3, x4, x5],                                    # x
                    [0.0, 1.0, y2, y3, y4, y5],                                     # y
                    [0.0, 0.0, 1.0, heading3, heading4, heading5],                  # heading
                    [0.0, 0.0, 0.0, 1.0 - (6.17*dt/15.2), 0.0, 0.0],                # vx
                    [0.0, 0.0, 0.0, vy3, vy4, vy5],                                 # vy
                    [0.0, 0.0, 0.0, omega3, omega4, omega5]])                        # omega
        return F

    def fmatrix2(self, state, control, dt):
        x,y,heading,vx,vy,omega = np.array(state).flatten()
        steering, throttle = np.array(control).flatten()

        x2 = dt*(-(dt*(6.17*throttle - 0.405921052631579*vx - 2.05461) + vx)*sin(heading) - (dt*(-1.0*omega*vx + 5.00192*sin(1.6*atan(2.3*steering - 2.3*atan((0.04824*omega + vy)/vx)))*cos(steering) + 6.644792*sin(1.6*atan(2.3*atan((0.04176*omega - vy)/vx)))) + vy)*cos(heading))
        x3 = dt*(-dt*(-1.0*omega + 18.4070656*(0.04824*omega + vy)*cos(steering)*cos(1.6*atan(2.3*steering - 2.3*atan((0.04824*omega + vy)/vx)))/(vx**2*(1 + (0.04824*omega + vy)**2/vx**2)*(5.29*(steering - atan((0.04824*omega + vy)/vx))**2 + 1)) - 24.45283456*(0.04176*omega - vy)*cos(1.6*atan(2.3*atan((0.04176*omega - vy)/vx)))/(vx**2*(1 + (0.04176*omega - vy)**2/vx**2)*(5.29*atan((0.04176*omega - vy)/vx)**2 + 1)))*sin(heading) + (1 - 0.405921052631579*dt)*cos(heading))
        x4 = -dt*(dt*(-18.4070656*cos(steering)*cos(1.6*atan(2.3*steering - 2.3*atan((0.04824*omega + vy)/vx)))/(vx*(1 + (0.04824*omega + vy)**2/vx**2)*(5.29*(steering - atan((0.04824*omega + vy)/vx))**2 + 1)) - 24.45283456*cos(1.6*atan(2.3*atan((0.04176*omega - vy)/vx)))/(vx*(1 + (0.04176*omega - vy)**2/vx**2)*(5.29*atan((0.04176*omega - vy)/vx)**2 + 1))) + 1)*sin(heading)
        x5 = -dt**2*(-1.0*vx - 0.887956844544*cos(steering)*cos(1.6*atan(2.3*steering - 2.3*atan((0.04824*omega + vy)/vx)))/(vx*(1 + (0.04824*omega + vy)**2/vx**2)*(5.29*(steering - atan((0.04824*omega + vy)/vx))**2 + 1)) + 1.0211503712256*cos(1.6*atan(2.3*atan((0.04176*omega - vy)/vx)))/(vx*(1 + (0.04176*omega - vy)**2/vx**2)*(5.29*atan((0.04176*omega - vy)/vx)**2 + 1)))*sin(heading)

        y2 = dt*((dt*(6.17*throttle - 0.405921052631579*vx - 2.05461) + vx)*cos(heading) - (dt*(-1.0*omega*vx + 5.00192*sin(1.6*atan(2.3*steering - 2.3*atan((0.04824*omega + vy)/vx)))*cos(steering) + 6.644792*sin(1.6*atan(2.3*atan((0.04176*omega - vy)/vx)))) + vy)*sin(heading))
        y3 = dt*(dt*(-1.0*omega + 18.4070656*(0.04824*omega + vy)*cos(steering)*cos(1.6*atan(2.3*steering - 2.3*atan((0.04824*omega + vy)/vx)))/(vx**2*(1 + (0.04824*omega + vy)**2/vx**2)*(5.29*(steering - atan((0.04824*omega + vy)/vx))**2 + 1)) - 24.45283456*(0.04176*omega - vy)*cos(1.6*atan(2.3*atan((0.04176*omega - vy)/vx)))/(vx**2*(1 + (0.04176*omega - vy)**2/vx**2)*(5.29*atan((0.04176*omega - vy)/vx)**2 + 1)))*cos(heading) + (1 - 0.405921052631579*dt)*sin(heading))
        y4 = dt*(dt*(-18.4070656*cos(steering)*cos(1.6*atan(2.3*steering - 2.3*atan((0.04824*omega + vy)/vx)))/(vx*(1 + (0.04824*omega + vy)**2/vx**2)*(5.29*(steering - atan((0.04824*omega + vy)/vx))**2 + 1)) - 24.45283456*cos(1.6*atan(2.3*atan((0.04176*omega - vy)/vx)))/(vx*(1 + (0.04176*omega - vy)**2/vx**2)*(5.29*atan((0.04176*omega - vy)/vx)**2 + 1))) + 1)*cos(heading)
        y5 = dt**2*(-1.0*vx - 0.887956844544*cos(steering)*cos(1.6*atan(2.3*steering - 2.3*atan((0.04824*omega + vy)/vx)))/(vx*(1 + (0.04824*omega + vy)**2/vx**2)*(5.29*(steering - atan((0.04824*omega + vy)/vx))**2 + 1)) + 1.0211503712256*cos(1.6*atan(2.3*atan((0.04176*omega - vy)/vx)))/(vx*(1 + (0.04176*omega - vy)**2/vx**2)*(5.29*atan((0.04176*omega - vy)/vx)**2 + 1)))*cos(heading)

        heading3 = dt**2*(177.163286295005*(0.04824*omega + vy)*cos(steering)*cos(1.6*atan(2.3*steering - 2.3*atan((0.04824*omega + vy)/vx)))/(vx**2*(1 + (0.04824*omega + vy)**2/vx**2)*(5.29*(steering - atan((0.04824*omega + vy)/vx))**2 + 1)) + 203.737779239256*(0.04176*omega - vy)*cos(1.6*atan(2.3*atan((0.04176*omega - vy)/vx)))/(vx**2*(1 + (0.04176*omega - vy)**2/vx**2)*(5.29*atan((0.04176*omega - vy)/vx)**2 + 1))) + dt**2*(354.32657259001*(0.04824*omega + vy)*cos(steering)*cos(1.6*atan(2.3*steering - 2.3*atan((0.04824*omega + vy)/vx)))/(vx**2*(1 + (0.04824*omega + vy)**2/vx**2)*(5.29*(steering - atan((0.04824*omega + vy)/vx))**2 + 1)) + 407.475558478511*(0.04176*omega - vy)*cos(1.6*atan(2.3*atan((0.04176*omega - vy)/vx)))/(vx**2*(1 + (0.04176*omega - vy)**2/vx**2)*(5.29*atan((0.04176*omega - vy)/vx)**2 + 1)))
        heading4 = dt**2*(-354.32657259001*cos(steering)*cos(1.6*atan(2.3*steering - 2.3*atan((0.04824*omega + vy)/vx)))/(vx*(1 + (0.04824*omega + vy)**2/vx**2)*(5.29*(steering - atan((0.04824*omega + vy)/vx))**2 + 1)) + 407.475558478511*cos(1.6*atan(2.3*atan((0.04176*omega - vy)/vx)))/(vx*(1 + (0.04176*omega - vy)**2/vx**2)*(5.29*atan((0.04176*omega - vy)/vx)**2 + 1))) + dt**2*(-177.163286295005*cos(steering)*cos(1.6*atan(2.3*steering - 2.3*atan((0.04824*omega + vy)/vx)))/(vx*(1 + (0.04824*omega + vy)**2/vx**2)*(5.29*(steering - atan((0.04824*omega + vy)/vx))**2 + 1)) + 203.737779239256*cos(1.6*atan(2.3*atan((0.04176*omega - vy)/vx)))/(vx*(1 + (0.04176*omega - vy)**2/vx**2)*(5.29*atan((0.04176*omega - vy)/vx)**2 + 1)))
        heading5 = dt**2*(-8.54635693087104*cos(steering)*cos(1.6*atan(2.3*steering - 2.3*atan((0.04824*omega + vy)/vx)))/(vx*(1 + (0.04824*omega + vy)**2/vx**2)*(5.29*(steering - atan((0.04824*omega + vy)/vx))**2 + 1)) - 8.50808966103132*cos(1.6*atan(2.3*atan((0.04176*omega - vy)/vx)))/(vx*(1 + (0.04176*omega - vy)**2/vx**2)*(5.29*atan((0.04176*omega - vy)/vx)**2 + 1))) + dt*(dt*(-17.0927138617421*cos(steering)*cos(1.6*atan(2.3*steering - 2.3*atan((0.04824*omega + vy)/vx)))/(vx*(1 + (0.04824*omega + vy)**2/vx**2)*(5.29*(steering - atan((0.04824*omega + vy)/vx))**2 + 1)) - 17.0161793220626*cos(1.6*atan(2.3*atan((0.04176*omega - vy)/vx)))/(vx*(1 + (0.04176*omega - vy)**2/vx**2)*(5.29*atan((0.04176*omega - vy)/vx)**2 + 1))) + 1)

        vy3 = dt*(-1.0*omega + 18.4070656*(0.04824*omega + vy)*cos(steering)*cos(1.6*atan(2.3*steering - 2.3*atan((0.04824*omega + vy)/vx)))/(vx**2*(1 + (0.04824*omega + vy)**2/vx**2)*(5.29*(steering - atan((0.04824*omega + vy)/vx))**2 + 1)) - 24.45283456*(0.04176*omega - vy)*cos(1.6*atan(2.3*atan((0.04176*omega - vy)/vx)))/(vx**2*(1 + (0.04176*omega - vy)**2/vx**2)*(5.29*atan((0.04176*omega - vy)/vx)**2 + 1)))
        vy4 = dt*(-18.4070656*cos(steering)*cos(1.6*atan(2.3*steering - 2.3*atan((0.04824*omega + vy)/vx)))/(vx*(1 + (0.04824*omega + vy)**2/vx**2)*(5.29*(steering - atan((0.04824*omega + vy)/vx))**2 + 1)) - 24.45283456*cos(1.6*atan(2.3*atan((0.04176*omega - vy)/vx)))/(vx*(1 + (0.04176*omega - vy)**2/vx**2)*(5.29*atan((0.04176*omega - vy)/vx)**2 + 1))) + 1
        vy5 = dt*(-1.0*vx - 0.887956844544*cos(steering)*cos(1.6*atan(2.3*steering - 2.3*atan((0.04824*omega + vy)/vx)))/(vx*(1 + (0.04824*omega + vy)**2/vx**2)*(5.29*(steering - atan((0.04824*omega + vy)/vx))**2 + 1)) + 1.0211503712256*cos(1.6*atan(2.3*atan((0.04176*omega - vy)/vx)))/(vx*(1 + (0.04176*omega - vy)**2/vx**2)*(5.29*atan((0.04176*omega - vy)/vx)**2 + 1)))

        omega3 = dt*(354.32657259001*(0.04824*omega + vy)*cos(steering)*cos(1.6*atan(2.3*steering - 2.3*atan((0.04824*omega + vy)/vx)))/(vx**2*(1 + (0.04824*omega + vy)**2/vx**2)*(5.29*(steering - atan((0.04824*omega + vy)/vx))**2 + 1)) + 407.475558478511*(0.04176*omega - vy)*cos(1.6*atan(2.3*atan((0.04176*omega - vy)/vx)))/(vx**2*(1 + (0.04176*omega - vy)**2/vx**2)*(5.29*atan((0.04176*omega - vy)/vx)**2 + 1)))
        omega4 = dt*(-354.32657259001*cos(steering)*cos(1.6*atan(2.3*steering - 2.3*atan((0.04824*omega + vy)/vx)))/(vx*(1 + (0.04824*omega + vy)**2/vx**2)*(5.29*(steering - atan((0.04824*omega + vy)/vx))**2 + 1)) + 407.475558478511*cos(1.6*atan(2.3*atan((0.04176*omega - vy)/vx)))/(vx*(1 + (0.04176*omega - vy)**2/vx**2)*(5.29*atan((0.04176*omega - vy)/vx)**2 + 1)))
        omega5 = dt*(-17.0927138617421*cos(steering)*cos(1.6*atan(2.3*steering - 2.3*atan((0.04824*omega + vy)/vx)))/(vx*(1 + (0.04824*omega + vy)**2/vx**2)*(5.29*(steering - atan((0.04824*omega + vy)/vx))**2 + 1)) - 17.0161793220626*cos(1.6*atan(2.3*atan((0.04176*omega - vy)/vx)))/(vx*(1 + (0.04176*omega - vy)**2/vx**2)*(5.29*atan((0.04176*omega - vy)/vx)**2 + 1))) + 1

        F = np.array([[1.0, 0.0, x2, x3, x4, x5],                                    # x
                    [0.0, 1.0, y2, y3, y4, y5],                                     # y
                    [0.0, 0.0, 1.0, heading3, heading4, heading5],                  # heading
                    [0.0, 0.0, 0.0, 1.0 - (6.17*dt/15.2), 0.0, 0.0],                # vx
                    [0.0, 0.0, 0.0, vy3, vy4, vy5],                                 # vy
                    [0.0, 0.0, 0.0, omega3, omega4, omega5]])                        # omega
        return F

    def b(self, state, control, dt):
        x,y,heading,vx,vy,omega = np.array(state).flatten()
        steering, throttle = np.array(control).flatten()

        x_steering = -dt**2*(-4.5472*sin(steering)*sin(1.6*atan(2.3*steering - 2.3*atan((0.04824*omega + vy)/vx))) + 16.733696*cos(steering)*cos(1.6*atan(2.3*steering - 2.3*atan((0.04824*omega + vy)/vx)))/(5.29*(steering - atan((0.04824*omega + vy)/vx))**2 + 1))*sin(heading)
        y_steering = dt**2*(-4.5472*sin(steering)*sin(1.6*atan(2.3*steering - 2.3*atan((0.04824*omega + vy)/vx))) + 16.733696*cos(steering)*cos(1.6*atan(2.3*steering - 2.3*atan((0.04824*omega + vy)/vx)))/(5.29*(steering - atan((0.04824*omega + vy)/vx))**2 + 1))*cos(heading)
        heading_steering = dt**2*(-87.5312679323147*sin(steering)*sin(1.6*atan(2.3*steering - 2.3*atan((0.04824*omega + vy)/vx))) + 322.115065990918*cos(steering)*cos(1.6*atan(2.3*steering - 2.3*atan((0.04824*omega + vy)/vx)))/(5.29*(steering - atan((0.04824*omega + vy)/vx))**2 + 1)) + dt**2*(-43.7656339661574*sin(steering)*sin(1.6*atan(2.3*steering - 2.3*atan((0.04824*omega + vy)/vx))) + 161.057532995459*cos(steering)*cos(1.6*atan(2.3*steering - 2.3*atan((0.04824*omega + vy)/vx)))/(5.29*(steering - atan((0.04824*omega + vy)/vx))**2 + 1))
        vy_steering = dt*(-4.5472*sin(steering)*sin(1.6*atan(2.3*steering - 2.3*atan((0.04824*omega + vy)/vx))) + 16.733696*cos(steering)*cos(1.6*atan(2.3*steering - 2.3*atan((0.04824*omega + vy)/vx)))/(5.29*(steering - atan((0.04824*omega + vy)/vx))**2 + 1))
        omega_steering = dt*(-87.5312679323147*sin(steering)*sin(1.6*atan(2.3*steering - 2.3*atan((0.04824*omega + vy)/vx))) + 322.115065990918*cos(steering)*cos(1.6*atan(2.3*steering - 2.3*atan((0.04824*omega + vy)/vx)))/(5.29*(steering - atan((0.04824*omega + vy)/vx))**2 + 1))

        B = np.array([[6.17*dt**2*cos(heading), x_steering],                         # x
                    [6.17*dt**2*sin(heading), y_steering],                          # y
                    [0.0, heading_steering],                                        # heading
                    [6.17*dt, 0.0],                                                 # vx
                    [0.0, vy_steering],                                             # vy
                    [0.0, omega_steering]])                                          # omega
        return B

    # Jacobian of dynamics f(x,u), df/dx, df/du
    # TODO calculate this
    def df(self, x,u):
        dfdx = np.zeros((self.n,self.n))
        dfdu = np.zeros((self.n,self.m))
        return dfdx, dfdu
    pass

# class Dynamics:
#     def __init__(self,car):
#         self.n = 6
#         self.m = 2
#         self.car = car
#         return

#     def tireCurve(self,slip):
#         C = 1.6
#         B = 2.3
#         D = 1.1
#         # C: tail shape
#         retval = D * np.sin( C * np.arctan(B *slip))
#         return retval

#     def f(self, state, control):
#         lf = self.car.lf
#         lr = self.car.lr
#         L = self.car.L

#         Iz = self.car.Iz
#         m = self.car.m

#         # NOTE here vx = vf, vy = vs, different convention
#         x,y,heading,vx,vy,omega = np.array(state).flatten()
#         steering, throttle = np.array(control).flatten()

#         # for small longitudinal velocity use kinematic model
#         if (vx<0.05):
#             beta = atan(lr/L*tan(steering))
#             norm = lambda a,b:(a**2+b**2)**0.5
#             # motor model
#             d_vx = 6.17*(throttle - vx/15.2 -0.333)
#             d_vy = 0
#             d_omega = 0.0

#         else:
#             slip_f = -np.arctan((omega*lf + vy)/vx) + steering
#             slip_r = np.arctan((omega*lr - vy)/vx)

#             Ffy = self.tireCurve(slip_f) * m * 9.8 *lr/(lr+lf)
#             Fry = 1.15*self.tireCurve(slip_r) * m * 9.8 *lf/(lr+lf)

#             # Dynamics
#             #d_vx = 1.0/m * (Frx - Ffy * np.sin( steering ) + m * vy * omega)
#             d_vx = 6.17*(throttle - vx/15.2 -0.333)
#             d_vy = 1.0/m * (Fry + Ffy * np.cos( steering ) - m * vx * omega)
#             d_omega = 1.0/Iz * (Ffy * lf * np.cos( steering ) - Fry * lr)

#         # back to global frame
#         vxg = vx*cos(heading)-vy*sin(heading)
#         vyg = vx*sin(heading)+vy*cos(heading)

#         return np.array([vxg, vyg, omega, d_vx, d_vy, d_omega])

#     # Jacobian of dynamics f(x,u), df/dx, df/du
#     # TODO calculate this
#     def df(self, x,u):
#         dfdx = np.zeros((self.n,self.n))
#         dfdu = np.zeros((self.n,self.m))
#         return dfdx, dfdu