from common import *
from math import isnan,pi,degrees,radians,sin,cos
from controller.CarController import CarController
from controller.PidController import PidController
from third_party.solve_lq_game import solve_lq_game
from extension.simulator.CurvilinearSimulator import CurvilinearSimulator

class DdpPointMaxxCarController(CarController):
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

        return

    def update_dynamics(self,states,controls,dt=None):
        if (dt is None):
            dt = self.dt
        return self.simulator.advancePointMassDynamics(states,controls,dt)

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
        lu = u.T @ R
        luu = R
        lux = 0
        return (lx,lxx,lu,luu,lux)


    def ddpControl(self,x0):
        # get reference u_ref
        if (self.u_ref is None):
            self.u_ref = np.zeros((self.horizon, self.m))
        def newState():
            return {'x':[x0]
                'u' : [],
                'fx' : [],
                'fu' : [],
                'lx' : [],
                'lxx' : [],
                'lu' : [],
                'luu' : [],
                'lux' : [],
                'Qx' : [],
                'Qu' : [],
                'Qux' : [],
                'Qxx' : [],
                'Quu' : [],
                'Vx' : [],
                'Vxx' : []
                }
        def addState(state,x,u,fx,fu,lx,lxx,lu,luu,lux):
            state['x'].append(x)
            state['u'].append(u)
            state['fx'].append(fx)
            state['fu'].append(fu)
            state['lx'].append(lx)
            state['lxx'].append(lxx)
            state['lu'].append(lu)
            state['luu'].append(luu)
            state['lux'].append(lux)
            return

                

        num_iter = 3
        for iter in range(num_iter):
            Vx = 0
            Vxx = 0
            xx =[x0]

            # rollout u
            # calculate derivatives for l(x,u) and f(x,u)
            for t in range(self.horizon):
                u = -np.linalg.inv(Quu_k[t]) @ (Qu_k[t].T + Qux_k[t] @ (xx[-1] - self.x_ref[t])) + self.u_ref[t]
                new_x = self.update_dynamics(xx[-1],u).reshape(-1,1)
                A,B,d = self.linearize(xx[-1], u)
                lx,lxx,lu,luu,lux = self.getLder(xx[-1], u)
                addState(state,x,u,fx,fu,lx,lxx,lu,luu,lux)

            for k in range(self.horizion,0,-1):
                print('step ', k)
                # calculate dQ_?
                # calculate dV_


        # roll out u_ref, get x_ref
        # linearize around _ref, get A,B,d
        xx =[x0]
        As = []
        Bs = []
        ds = []
        for t in range(self.horizon):
            # x+ = A x + B u + d, for x~x_ref
            # ~x = x - x_ref
            # ~x+ = A~x + B~u
            new_x = self.update_dynamics(xx[-1],self.u_ref[t])
            A,B,d = self.linearize(xx[-1], self.u_ref[t])

            xx.append(new_x)
            As.append(A)
            Bs.append(B)
            ds.append(d)
        

    # differentiate dynamics around nominal state and control
    # return: A, B, d, s.t. x_k+1 = Ax + Bu + d
    def linearize(self, nominal_state, nominal_ctrl):
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

