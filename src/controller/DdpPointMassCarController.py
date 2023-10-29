from common import *
from math import isnan,pi,degrees,radians,sin,cos
from controller.CarController import CarController
from controller.PidController import PidController
from extension.simulator.CurvilinearSimulator import CurvilinearSimulator

class DdpPointMassCarController(CarController):
    def __init__(self, car,config):
        super().__init__(car,config)
        self.m = 2
        self.n = 4

        self.u_ref = None
        self.horizion = 10
        self.dt = self.main.dt

    def init(self):
        self.simulator = self.main.simulator
        assert(isinstance(self.simulator,CurvilinearSimulator))

    def control(self):
        for car in self.main.cars:
            # s,v,n,phi
            #throttle = 1.0 if car.sim_states[1] < 1.0 else -1.0
            #steering = -car.sim_states[3] - car.sim_states[2]
            #print(car.sim_states)
            #print(f'T = {throttle} S = {steering}')

            steering,throttle = self.ddpControl(car.sim_states)

            car.throttle = throttle
            car.steering = steering
        self.drawPredictedTrajectory()
        return

    def update_dynamics(self,states,controls,dt=None):
        if (dt is None):
            dt = self.dt
        return self.simulator.advancePointMassDynamics(states.flatten(),controls.flatten(),dt)

    def getLder(self,x_ref, u_ref):
        ''' 
        jacobian and hessian matrix for the step cost l(x,u) 
        l(x,u) = xT Q x + q x + uT R u
        [x_ref]: np array size n*1
        [u_ref]: np array size m*1
        '''
        Kn = 1.0
        Kphi = 1.0
        Ks = 1.0
        Kv = 1.0
        Q = np.diag([0,0,Kn,Kphi])
        q = np.array([[-Ks, -Kv,0,0]])
        R = np.diag([1.0,1.0])
        lx = x_ref.T @ Q + q
        lxx = Q
        lu = u_ref.T @ R
        luu = R
        lux = 0
        return (lx,lxx,lu,luu,lux)


    def ddpControl(self,x0):
        # get reference u_ref
        if (self.u_ref is None):
            self.u_ref = np.zeros((self.horizon, self.m,1))
            self.u_ref[:,1,:] = 1.0
            self.x_ref = [x0.reshape(self.n,1)]
            for t in range(self.horizon):
                new_x = self.update_dynamics(self.x_ref[t],self.u_ref[t]).reshape(-1,1)
                self.x_ref.append(new_x)

        u_ref = self.u_ref
        x_ref = self.x_ref
        u_forward_vec = [np.zeros((self.m,1))] * self.horizon
        u_feedback_K_vec = [np.zeros((self.m, self.n))] * self.horizon

        num_iter = 3
        for iter in range(num_iter):
            xx = [x0.reshape(self.n,1)]
            uu = []

            # rollout u, forward propagate
            # calculate derivatives for l(x,u) and f(x,u)
            for t in range(self.horizon):
                u = u_forward_vec[t] + u_feedback_K_vec[t] @ (xx[-1]-x_ref[t])
                + u_ref[t]
                new_x = self.update_dynamics(xx[-1],u).reshape(-1,1)
                xx.append(new_x)
                uu.append(u)
            x_ref = xx
            u_ref = uu

            # V(self.horizon+1) = 0
            Vx = np.zeros((1,self.n))
            Vxx = np.zeros((self.n,self.n))
            # backward propagate, get V, Q, feedforward and feedback control
            # u = u_forward + u_feedback_K_vec @ (x-x_ref) + u_ref
            for k in range(self.horizion,0,-1):
                fx,fu,d = self.linearize(xx[k], uu[k])
                lx,lxx,lu,luu,lux = self.getLder(xx[k], uu[k])

                Qx = lx + Vx @ fx
                Qu = lu + Vx @ fu
                Qxx = lxx + fx.T @ Vxx @ fx # dropping Vx fxx dx
                Quu = luu + fu.T @ Vxx @ fu
                Qux = lux + fu.T @ Vxx @ fx

                u_forward = -np.linalg.inv(Quu) @ Qu.T
                u_feedback_K = -np.linalg.inv(Quu) @ Qux
                u_forward_vec.insert(0,u_forward)
                u_feedback_K_vec.insert(0,u_feedback_K)

                Vx = Qx - Qu @ np.linalg.inv(Quu) @ Qux
                Vxx = Qxx - Qux.T @ np.linalg.inv(Quu) @ Qux

        self.u_ref = u_ref
        self.x_ref = x_ref
        return u_ref[0].flatten()


    def linearize(self, nominal_state, nominal_ctrl):
        '''
        differentiate dynamics around nominal state and control
        return: A, B, d, s.t. x_k+1 = Ax + Bu + d
        '''
        nominal_state = np.array(nominal_state).copy()
        nominal_ctrl = np.array(nominal_ctrl).copy()
        epsilon = 1e-2

        # A = df/dx
        A = np.zeros((self.n,self.n),dtype=np.float)
        # find A
        for i in range(self.n):
            # d x / d x_i, ith row in A
            x_l = nominal_state.copy()
            x_l[i] -= epsilon

            x_post_l = self.update_dynamics(x_l, nominal_ctrl, self.dt)

            x_r = nominal_state.copy()
            x_r[i] += epsilon
            x_post_r = self.update_dynamics(x_r, nominal_ctrl, self.dt)

            A[:,i] += (x_post_r.flatten() - x_post_l.flatten()) / (2*epsilon)
            '''
            print("perturbing x%d"%(i))
            print(A[:,i])
            breakpoint()
            print("")
            '''


        # B = df/du
        B = np.zeros((self.n,self.m),dtype=np.float)
        # find B
        for i in range(self.m):
            # d x / d u_i, ith row in B
            x0 = nominal_state.copy()
            u_l = nominal_ctrl.copy()
            u_l[i] -= epsilon
            x_post_l = self.update_dynamics(x0, u_l, self.dt)
            x_post_l = x_post_l.copy()

            x0 = nominal_state.copy()
            u_r = nominal_ctrl.copy()
            u_r[i] += epsilon
            x_post_r = self.update_dynamics(x0, u_r, self.dt)
            x_post_r = x_post_r.copy()

            B[:,i] += (x_post_r.flatten() - x_post_l.flatten()) / (2*epsilon)

        x0 = nominal_state.copy()
        u0 = nominal_ctrl.copy()
        '''
        self.sim.states = np.array(x0.copy())
        self.sim.updateCar(self.dt,None,nominal_ctrl[0],nominal_ctrl[1])
        x_post = np.array(self.sim.states)
        '''
        x_post = self.update_dynamics(x0, u0, self.dt)


        # d = x_k+1 - Ak*x_k - Bk*u_k
        x0 = nominal_state.copy()
        u0 = nominal_ctrl.copy()
        d = x_post.flatten() - A @ x0 - B @ u0

        return A,B,d

    def drawPredictedTrajectory(self):
        ''' draw self.x_ref '''
        lineColor = (0,255,0)
        if (self.main.visualization.update_visualization.is_set()):
            img = self.main.visualization.visualization_img
            for (i,car) in enumerate(self.main.cars):
                predicted_traj = []
                for t in range(self.horizon):
                    curvi_states = self.x_ref[t]
                    control = self.u_ref[t]
                    cart_states = self.simulator.curv2Cart(curvi_states)
                    predicted_traj.append(cart_states)

                predicted_traj = np.array(predicted_traj)
                predicted_traj = np.hstack([np.zeros((predicted_traj.shape[0],1)), predicted_traj])
                img = self.main.track.drawTrajectory(np.array(predicted_traj),img,lineColor)
            self.main.visualization.visualization_img = img

