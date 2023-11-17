from common import *
from math import isnan,pi,degrees,radians,sin,cos
from controller.CarController import CarController
from controller.PidController import PidController
from third_party.solve_lq_game import solve_lq_game
from extension.simulator.CurvilinearSimulator import CurvilinearSimulator
from util.SymbolicDynamics import SymbolicDynamics
from scipy.linalg import block_diag
import sympy

class iLQGameCarController(CarController):
    def __init__(self, car,config):
        super().__init__(car,config)
        self.m = 2
        self.n = 4

        # for ego agent i
        # horizon*m*1
        self.u_ref = np.zeros((self.horizon, self.m,1))
        # horizon*n*1
        self.x_ref = np.zeros((self.horizon, self.n,1))

        # for ego agent j
        self.u_j_ref = np.zeros((self.horizon, self.m,1))
        self.x_j_ref = np.zeros((self.horizon, self.n,1))

        self.horizon = 20
        self.dt = self.main.dt
        # symbolic dynamics
        #self.sym = self.buildSymbolicDynamics()

        # cost to apply on state
        # state x: s,v,n,phi
        self.Q1 = np.diag([0,1.0,15.0,9.0])
        self.q1 = np.array([[-1.0, -3,0,0]]).T
        # cost on control u (ay,ax)
        self.R1 = np.diag([0.1,0.1])

        self.Q2 = np.diag([0,1.0,15.0,3.0])
        self.q2 = np.array([[-1.0, -4.5,0,0]]).T
        self.R2 = np.diag([0.1,0.1])

        # cost on track boundary
        self.boundary_min_distance = 0.06
        self.boundary_cost = 60.0

        # cost on opponent collision
        Kop = 30.0
        self.opponent_min_distance_s = 0.25
        self.opponent_min_distance_n = 0.17
        # this should be negative
        self.Qop = -np.diag([Kop,0,Kop,0])
        self.linearize_around_zero_control = True

    def init(self):
        if (self.linearize_around_zero_control):
            self.print_warning('----- Linearizing around u=0 ----- ')
        self.simulator = self.main.simulator
        assert(isinstance(self.simulator,CurvilinearSimulator))

    def control(self):
        self.debug_dict = {}
        assert(len(self.main.cars)==2)
        # s,v,n,phi
        ctrl0, ctrl1 = self.lqControl(self.main.cars[0].sim_states, self.main.cars[1].sim_states)
        print(f'car0: {self.main.cars[0].sim_states}, ctrl = {ctrl0}')
        print(f'car1: {self.main.cars[1].sim_states}, ctrl = {ctrl1}')
        # DEBUG
        delta_x = self.main.cars[0].sim_states - self.main.cars[1].sim_states 
        dist = (delta_x[0]**2 + delta_x[2]**2)**0.5
        print(f'dist: {dist}')

        self.main.cars[0].steering = ctrl0[0]
        self.main.cars[0].throttle = ctrl0[1]

        self.main.cars[1].steering = ctrl1[0]
        self.main.cars[1].throttle = ctrl1[1]

        self.drawPredictedTrajectory()
        return

    def buildSymbolicDynamics(self):
        sym = SymbolicDynamics(self.n,self.m)
        # curvature at current s
        k_s = sym.k_s = sympy.symbols('k_s')
        sym.xop = [sympy.symbols(f'xop{i}') for i in range(self.n)]
        s = sym.x[0]
        v = sym.x[1]
        n = sym.x[2]
        phi = sym.x[3]

        ay = sym.u[0]
        ax = sym.u[1]

        dsdt = v*sympy.cos(phi)/(1-n*k_s)
        dvdt = ax
        dndt = v*sympy.sin(phi)
        dphidt = ay/v - k_s*dsdt

        new_s = s + dsdt*self.dt
        new_v = v + dvdt*self.dt
        new_n = n + dndt*self.dt
        new_phi = phi + dphidt*self.dt

        sym.f = [new_s, new_v, new_n, new_phi]

        #l_path(x,u) = xT Q x + q x + uT R u
        #l_op(x,xop) = (x-xop)T Qop (x-xop) = (remove const) xT Qop x - 2xopT Qop x
        l_path = sym.xQx_diag(sym.x,self.Q) + sym.product(self.q, sym.x) + self.xQx_diag(sym.u, self.R)
        l_op = sym.xQx_diag(sym.minus(sym.x,sym.xop), self.Qop)
        sym.l = l_path + l_op
        sym.symDer()
        return sym

    def linearizeSymbolicDynamics(self,x0,u0,xop):
        sym = self.sym
        x0 = x0.flatten()
        u0 = u0.flatten()
        xop = xop.flatten()
        k_s = self.simulator.curvature(x0[0])
        subs_dict = {sym.k_s:k_s}
        for i in range(self.n):
            subs_dict.update({sym.xop[i]:xop[i]})

        fx,fu,lx,lu,lxx,luu,lux = self.sym.calcDer(x0=x0, u0=u0, subs_dict=subs_dict)
        return fx,fu,lx,lu,lxx,luu,lux

    def update_dynamics(self,states,controls,dt=None):
        if (dt is None):
            dt = self.dt
        return self.simulator.advancePointMassDynamics(states.flatten(),controls.flatten(),dt)

    # FIXME
    def evalCost(self,x0,uus, As,BBs, QQs, lls, RRs):
        ''' calculate cost for given initial state and control sequence '''
        u_i = uus[0]
        u_j = uus[1]
        B1s = BBs[0]
        B2s = BBs[1]
        Q1s = QQs[0]
        Q2s = QQs[1]
        l1s = lls[0]
        l2s = lls[1]
        R1 = RRs[0][0]
        R2 = RRs[1][1]

        xx = [x0]
        cost_i = 0
        cost_j = 0
        for t in range(self.horizon):
            # FIXME incorrect dynamics
            x = As[t] @ dx[t] + B1s[t] @ u_i + B2s[t] @ u_j
            cost_i += x.T @ Q1s[t] @ x + l1s[t] @ x + u_i[0] @ R1 @ u_i[0]
            cost_j += x.T @ Q2s[t] @ x + l2s[t] @ x + u_j[0] @ R2 @ u_j[0]
        return cost_i,cost_j



    def lqControl(self,x0_i,x0_j):
        alpha = 0.5
        P1s = [np.zeros((self.m,self.n*2))]*self.horizon
        P2s = [np.zeros((self.m,self.n*2))]*self.horizon
        alpha1s = [np.zeros((self.m,1))]*self.horizon
        alpha2s = [np.zeros((self.m,1))]*self.horizon
        if (self.linearize_around_zero_control):
            self.u_ref = np.zeros((self.horizon, self.m,1))
            self.u_j_ref = np.zeros((self.horizon, self.m,1))

        # iterations
        for p in range(3):
            # roll out u_ref, get x_ref
            # linearize around _ref, get A,B,d
            xx_i =[x0_i.reshape((self.n,1))]
            Ais = []
            Bis = []
            dis = []
            uu_i = []

            xx_j =[x0_j.reshape((self.n,1))]
            Ajs = []
            Bjs = []
            djs = []
            uu_j = []

            # u_ki = u_ref_ki - P_ki @ dx_k - alpha_ki
            for t in range(self.horizon):
                #for ego agent i
                # x+ = A x + B u + d, for x~x_ref
                # ~x = x - x_ref
                # ~x+ = A~x + B~u
                dx_i = xx_i[-1] - self.x_ref[t]
                dx_j = xx_j[-1] - self.x_j_ref[t]
                dx = np.vstack([dx_i,dx_j])

                #for ego agent i
                u = self.u_ref[t] - P1s[t] @ dx - alpha1s[t]
                new_x = self.update_dynamics(xx_i[-1],u)
                if (self.linearize_around_zero_control):
                    A,B,d = self.linearize(xx_i[-1],np.zeros(self.m))
                else:
                    A,B,d = self.linearize(xx_i[-1],u)
                xx_i.append(new_x.reshape(4,1))
                Ais.append(A)
                Bis.append(B)
                dis.append(d)
                uu_i.append(u)

                #for ego agent j
                u = self.u_j_ref[t] - P2s[t] @ dx - alpha2s[t]
                new_x = self.update_dynamics(xx_j[-1],u)
                if (self.linearize_around_zero_control):
                    A,B,d = self.linearize(xx_j[-1],np.zeros(self.m))
                else:
                    A,B,d = self.linearize(xx_j[-1],u)
                xx_j.append(new_x.reshape(4,1))
                Ajs.append(A)
                Bjs.append(B)
                djs.append(d)
                uu_j.append(u)

            self.x_ref = xx_i
            self.u_ref = uu_i
            self.x_j_ref = xx_j
            self.u_j_ref = uu_j
            n = self.n
            m = self.m

            As = [block_diag(Ais[i],Ajs[i]) for i in range(len(Ais))]
            B1s = [np.vstack([Bi,np.zeros((n,m))]) for Bi in Bis]
            B2s = [np.vstack([np.zeros((n,m)),Bj]) for Bj in Bjs]


            Q1s,q1s,Q2s,q2s,Rs = self.getCostMatrices(xx_i,uu_i,xx_j,uu_j)

            # LQ cost function, get Q,l, Rs
            [P1s, P2s], [alpha1s, alpha2s] = solve_lq_game(
                As, [B1s, B2s],
                [Q1s, Q2s], [q1s, q2s], Rs)

            # DEBUG compare "expected" states from LQ game against simulated states
            # reference u is zero
            '''
            dx = [np.zeros((2*n,1))]
            x_predicted = []
            u_predicted = []
            for t in range(self.horizon):
                u_i = - P1s[t] @ dx[t] - alpha1s[t]
                u_j = - P2s[t] @ dx[t] - alpha2s[t]
                dx_new = As[t] @ dx[t] + B1s[t] @ u_i + B2s[t] @ u_j
                dx.append(dx_new)

                x = np.vstack([self.x_ref[t], self.x_j_ref[t]])
                x_predicted.append(x+dx[t])
                u_predicted.append([ u_i, u_j ])
            # plot x_predicted
            x_predicted = np.array(x_predicted)
            self.drawTrajectory(traj=x_predicted[:,:4,0], lineColor=(0,0,100))
            self.drawTrajectory(traj=x_predicted[:,4:,0], lineColor=(0,0,100))
            '''
            # DEBUG evaluate cost for both agents

        #ctrl1 = self.u_ref[0].flatten()
        #ctrl2 = self.u_j_ref[0].flatten()
        ctrl1 = -alpha1s[0].flatten()
        ctrl2 = -alpha2s[0].flatten()

        self.debug_dict.update({'u_ref':np.array(self.u_ref), 'x_ref':np.array(self.x_ref), 'x_j_ref':np.array(self.x_j_ref), 'u_j_ref':np.array(self.u_j_ref)})
        return ctrl1,ctrl2

    def getCostMatrices(self,xx_i,uu_i,xx_j,uu_j):
        Q1s = []
        Q2s = []
        q1s = []
        q2s = []

        R11s = []
        R22s = []

        R12s = []
        R21s = []

        n = self.n
        m = self.m
        II = np.hstack([np.eye(n),-np.eye(n)])
        R0 = np.zeros((m,m))

        for t in range(self.horizon):
            # these work on the state x, not state perturbation dx
            # cost_i = x.T @ Qi_x @ x + qi_x.T @ x + ui.T @ R @ ui
            Q1_x =  block_diag(self.Q1,np.zeros((n,n)))
            q1_x = np.hstack([self.q1.T,np.zeros((1,n))])

            Q2_x =  block_diag(np.zeros((n,n)),self.Q2)
            q2_x = np.hstack([np.zeros((1,n)),self.q2.T])

            # barrier function: opponent collision
            delta_x = xx_i[t] - xx_j[t]
            if (np.abs(delta_x[0])<self.opponent_min_distance_s and np.abs(delta_x[2])<self.opponent_min_distance_n):
                Q1_x += II.T @ self.Qop @ II
                Q2_x += II.T @ self.Qop @ II

            # barrier function: track boundary
            cart_states_i = self.simulator.curv2Cart(xx_i[t].flatten())
            left_boundary_i, right_boundary_i = self.main.track.preciseTrackBoundary(cart_states_i[:2],cart_states_i[2])

            # n>0 -> left
            if (left_boundary_i < self.boundary_min_distance or right_boundary_i < self.boundary_min_distance):
                self.print_info('car i out of bounds')
                Q_bdry = np.diag([0,0,self.boundary_cost,0,0,0,0,0])
                Q1_x += Q_bdry

            cart_states_j = self.simulator.curv2Cart(xx_j[t].flatten())
            left_boundary_j, right_boundary_j = self.main.track.preciseTrackBoundary(cart_states_j[:2],cart_states_j[2])

            # n>0 -> left
            if (left_boundary_j < self.boundary_min_distance or right_boundary_j < self.boundary_min_distance):
                self.print_info('car j out of bounds')
                Q_bdry = np.diag([0,0,0,0,0,0,self.boundary_cost,0])
                Q2_x += Q_bdry


            # TODO
            # barrier function: control limit
            R1 = self.R1
            R2 = self.R1


            xx_ref = np.vstack([self.x_ref[0], self.x_j_ref[0]])
            # these work on state perturbation dx
            Q1 = Q1_x
            q1 = 2 * xx_ref.T @ Q1_x + q1_x

            Q2 = Q2_x
            q2 = 2 * xx_ref.T @ Q2_x + q2_x

            Q1s.append(Q1)
            Q2s.append(Q2)
            q1s.append(q1.T)
            q2s.append(q2.T)

            R11s.append(R1)
            R22s.append(R2)

            R12s.append(R0)
            R21s.append(R0)

        Rs = [[R11s, R12s], [R21s, R22s]]

        return Q1s,q1s,Q2s,q2s,Rs

    # differentiate dynamics around nominal state and control
    # return: A, B, d, s.t. x_k+1 = Ax + Bu + d
    def linearize(self, nominal_state, nominal_ctrl):
        nominal_state = np.array(nominal_state.flatten()).copy()
        nominal_ctrl = np.array(nominal_ctrl.flatten()).copy()
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

    def drawPredictedTrajectory(self, lineColor=(0,100,100)):
        self.drawTrajectory(self.x_ref, lineColor=(0,100,100))
        self.drawTrajectory(self.x_j_ref, lineColor=(0,100,100))
        return

    def drawTrajectory(self, traj=None, lineColor=(0,100,100)):
        #lineColor = (0x22,0x6C,0xFF)
        if (self.main.visualization.update_visualization.is_set()):
            img = self.main.visualization.visualization_img
            predicted_traj = []
            for t in range(self.horizon):
                curvi_states = traj[t]
                cart_states = self.simulator.curv2Cart(curvi_states)
                predicted_traj.append(cart_states)

            predicted_traj = np.array(predicted_traj)
            predicted_traj = np.hstack([np.zeros((predicted_traj.shape[0],1)), predicted_traj])
            img = self.main.track.drawTrajectory(np.array(predicted_traj),img,lineColor)
            self.main.visualization.visualization_img = img
