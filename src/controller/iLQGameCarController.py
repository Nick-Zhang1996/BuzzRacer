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

        # horizon*m*1
        self.u_ref = None
        # horizon*n*1
        self.x_ref = None

        self.u_j_ref = None
        self.x_j_ref = None
        self.horizon = 20
        self.dt = self.main.dt
        # symbolic dynamics
        #self.sym = self.buildSymbolicDynamics()

        # cost
        Kn = 13.0
        Kphi = 1.0
        Ks = 1.0
        Kv = 1.0 * 0
        Kop = 0.002
        self.Q = np.diag([0,0.03,Kn,Kphi])
        self.q = np.array([[-Ks, -Kv,0,0]]).T
        self.R = np.diag([0.1,0.1])
        # opponent collision
        self.Qop = -np.diag([Kop,0,Kop,0])

    def init(self):
        self.simulator = self.main.simulator
        assert(isinstance(self.simulator,CurvilinearSimulator))

    def control(self):
        assert(len(self.main.cars)==2)
        # s,v,n,phi
        ctrl0, ctrl1 = self.lqControl(self.main.cars[0].sim_states, self.main.cars[1].sim_states)
        print(f'car0: v={self.main.cars[0].sim_states[1]}, ctrl = {ctrl0}')
        print(f'car1: v={self.main.cars[1].sim_states[1]}, ctrl = {ctrl1}')

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


    def lqControl(self,x0_i,x0_j):

        # 1: use new open loop policy
        alpha = 0.5
        P1s = [np.zeros((self.m,self.n*2))]*self.horizon
        P2s = P1s
        alpha1s = [np.zeros((self.m,1))]*self.horizon
        alpha2s = alpha1s
        # keep doing until convergence
        #while (np.linalg.norm(10)>0.01):
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

            if (self.u_ref is None):
                # first iteration, rollout u_ref, get x_ref
                # for ego agent i
                self.u_ref = np.zeros((self.horizon, self.m,1))
                # for ego agent j
                self.u_j_ref = np.zeros((self.horizon, self.m,1))
                for t in range(self.horizon):
                    u = self.u_ref[t]
                    new_x = self.update_dynamics(xx_i[-1],u)
                    A,B,d = self.linearize(xx_i[-1],u)
                    # DEBUG test symbolic differentiation

                    xx_i.append(new_x.reshape(4,1))
                    Ais.append(A)
                    Bis.append(B)
                    dis.append(d)
                    uu_i.append(u)

                    u = self.u_j_ref[t]
                    new_x = self.update_dynamics(xx_j[-1],u)
                    A,B,d = self.linearize(xx_j[-1],u)
                    # DEBUG test symbolic differentiation

                    xx_j.append(new_x.reshape(4,1))
                    Ajs.append(A)
                    Bjs.append(B)
                    djs.append(d)
                    uu_j.append(u)
            else:
                # 2+ iteration, apply feedback control law
                # u_ki = u_ref_ki = P_ki @ dx_k - alpha_ki
                for t in range(self.horizon):
                    #for ego agent i
                    # x+ = A x + B u + d, for x~x_ref
                    # ~x = x - x_ref
                    # ~x+ = A~x + B~u
                    dx_i = xx_i[-1] - self.x_ref[t]
                    dx_j = xx_j[-1] - self.x_j_ref[t]
                    dx = np.vstack([dx_i,dx_j])

                    u = self.u_ref[t] - P1s[t] @ dx - alpha1s[t]
                    new_x = self.update_dynamics(xx_i[-1],u)
                    A,B,d = self.linearize(xx_i[-1], u)
                    xx_i.append(new_x.reshape(4,1))
                    Ais.append(A)
                    Bis.append(B)
                    dis.append(d)
                    uu_i.append(u)

                    #for ego agent j
                    u = self.u_j_ref[t] - P2s[t] @ dx - alpha2s[t]
                    new_x = self.update_dynamics(xx_j[-1],u)
                    A,B,d = self.linearize(xx_j[-1], u)
                    xx_j.append(new_x.reshape(4,1))
                    Ajs.append(A)
                    Bjs.append(B)
                    djs.append(d)
                    uu_j.append(u)

            # TODO: check linearization, does ABd give correct result

            self.x_ref = xx_i
            self.u_ref = uu_i
            self.x_j_ref = xx_j
            self.u_j_ref = uu_j
            n = self.n
            m = self.m

            As = [block_diag(Ais[i],Ajs[i]) for i in range(len(Ais))]
            B1s = [np.vstack([Bi,np.zeros((n,m))]) for Bi in Bis]
            B2s = [np.vstack([np.zeros((n,m)),Bj]) for Bj in Bjs]

            II = np.hstack([np.eye(n),-np.eye(n)])
            Q1 =  block_diag(self.Q,np.zeros((n,n))) + II.T @ self.Qop @ II
            q1 = np.hstack([self.q.T,np.zeros((1,n))])
            R1 = self.R

            # TODO: check cost approximation
            Q2 =  block_diag(np.zeros((n,n)),self.Q) + II.T @ self.Qop @ II
            q2 = np.hstack([np.zeros((1,n)),self.q.T])
            R2 = self.R


            Q1s = [Q1]*self.horizon
            Q2s = [Q2]*self.horizon
            l1s = [q1.T]*self.horizon
            l2s = [q2.T]*self.horizon

            R0 = np.zeros((m,m))
            R11s = [R1]*self.horizon
            R22s = [R2]*self.horizon

            R12s = [R0]*self.horizon
            R21s = [R0]*self.horizon

            # LQ cost function, get Q,l, Rs
            [P1s, P2s], [alpha1s, alpha2s] = solve_lq_game(
                As, [B1s, B2s],
                [Q1s, Q2s], [l1s, l2s], [[R11s, R12s], [R21s, R22s]])
            # list of size horizon, each element is of dim m*1
            #print('alpha: ',alpha1s,alpha2s)
            #print(np.linalg.norm(np.array(alpha1s)+np.array(alpha2s)))

        ctrl1 = self.u_ref[0].flatten()
        ctrl2 = self.u_j_ref[0].flatten()
        return ctrl1,ctrl2

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

    def drawPredictedTrajectory(self):
        ''' draw self.x_ref '''
        lineColor = (0,100,100)
        if (self.main.visualization.update_visualization.is_set()):
            img = self.main.visualization.visualization_img
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

        ''' draw self.x_j_ref '''
        lineColor = (0x22,0x6C,0xFF)
        if (self.main.visualization.update_visualization.is_set()):
            img = self.main.visualization.visualization_img
            predicted_traj = []
            for t in range(self.horizon):
                curvi_states = self.x_j_ref[t]
                control = self.u_j_ref[t]
                cart_states = self.simulator.curv2Cart(curvi_states)
                predicted_traj.append(cart_states)

            predicted_traj = np.array(predicted_traj)
            predicted_traj = np.hstack([np.zeros((predicted_traj.shape[0],1)), predicted_traj])
            img = self.main.track.drawTrajectory(np.array(predicted_traj),img,lineColor)
            self.main.visualization.visualization_img = img

