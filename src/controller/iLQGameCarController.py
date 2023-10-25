from common import *
from math import isnan,pi,degrees,radians,sin,cos
from controller.CarController import CarController
from controller.PidController import PidController
from third_party.solve_lq_game import solve_lq_game
from extension.simulator.CurvilinearSimulator import CurvilinearSimulator

class iLQGameCarController(CarController):
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

            steering,throttle = self.lqControl(car.sim_states)

            car.throttle = throttle
            car.steering = steering

        return

    def update_dynamics(self,states,controls,dt=None):
        if (dt is None):
            dt = self.dt
        return self.simulator.advancePointMassDynamics(states,controls,dt)


    def lqControl(self,x0):
        # get reference u_ref
        if (self.u_ref is None):
            self.u_ref = np.zeros((self.horizon, self.m))

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
        
        #xx = np.array(xx)
        #As = np.array(As)
        #Bs = np.array(Bs)
        #ds = np.array(ds)

        Q = np.diag([0,0,1,1])
        Q1s = [Q]*self.horizon
        Q2s = [Q]*self.horizon
        #l = np.zeros((self.n,1))
        l = np.array([[-0.5,-0.5,0,0]]).T
        l1s = [l]*self.horizon
        l2s = [l]*self.horizon

        R = np.diag([0.1,0.1])
        R0 = np.zeros((self.m,self.m))
        R11s = [R]*self.horizon
        R22s = [R]*self.horizon

        R12s = [R0]*self.horizon
        R21s = [R0]*self.horizon

        B1s = Bs
        B2s = Bs

        # LQ cost function, get Q,l, Rs
        [P1s, P2s], [alpha1s, alpha2s] = solve_lq_game(
            As, [B1s, B2s],
            [Q1s, Q2s], [l1s, l2s], [[R11s, R12s], [R21s, R22s]])

        # list of size horizon, each element is of dim m*1
        #print('alpha: ',alpha1s,alpha2s)

        ctrl = alpha1s[0].flatten()

        # TODO generate new u_ref
        return ctrl

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
