from common import *
from math import isnan,pi,degrees,radians,sin,cos
from controller.CarController import CarController
from controller.PidController import PidController
from third_party.solve_lq_game import solve_lq_game
from controller.LQGame import my_solve_lq_game
from extension.simulator.CurvilinearSimulator import CurvilinearSimulator
from util.SymbolicDynamics import SymbolicDynamics
from scipy.linalg import block_diag
import sympy
from util.timeUtil import execution_timer
import cv2

class iLQGameCarController(CarController):
    def __init__(self, car,config):
        super().__init__(car,config)
        self.t = execution_timer(True)
        self.lqt = execution_timer(True)
        self.m = 2
        self.n = 4
        self.iterations = 3
        self.draw_prediction = True
        self.debug = False
        self.leader_collision_ignorant = False

        # for ego agent i -> car 0
        # horizon*m*1
        self.u_i_ref = np.zeros((self.horizon, self.m,1))
        # horizon*n*1
        self.x_i_ref = np.zeros((self.horizon, self.n,1))

        # for ego agent j -> car 1
        self.u_j_ref = np.zeros((self.horizon, self.m,1))
        self.x_j_ref = np.zeros((self.horizon, self.n,1))

        self.horizon = 20
        self.dt = self.main.dt * 2
        # symbolic dynamics
        self.sym = self.buildSymbolicDynamics()

        # cost to apply on state
        # state x: s,v,n,phi
        self.Q1 = np.diag(  [ 0,0.00,1.0,1.0])
        self.q1 = np.array([[-4,0,0,0]]).T
        # aggressiveness: 0->don't care about opponent 1->J = s_i - s_j
        self.Qop1 = 0
        self.Qop1_leading = 0
        self.Qop1_following = -2
        self.Qop1_blocking = 0
        self.adaptive_Qop = False

        self.Q2 = np.diag(  [ 0,0.00,1.0,1.0])
        self.q2 = np.array([[-4,0,0,0]]).T
        self.Qop2 = 0

        # cost on track boundary
        #self.boundary_min_distance = 0.06 * 2
        self.boundary_min_distance = 0.02
        self.boundary_cost = 30.0*3

        # cost on opponent collision
        Kcol = 30.0*2
        self.opponent_min_distance_s = 0.3
        self.opponent_min_distance_n = 0.1
        self.Qcol = np.diag([Kcol,0,Kcol,0])

        # cost on control
        self.control_barrier_cost = 0.1*10
        self.circular_control_barrier = True
        self.linearize_around_zero_control = False
        # ratio of new control to use, 1->use new 0->use old
        self.alpha = 1.0
        # if true, this controller will control opponent
        self.control_opponent = False
        # if true, add another layer of optimization for ego agent (i)
        self.blocking_control = False
        ConfigObject.__init__(self,config)

    def preInit(self):
        self.createConstants()
        self.overrideControlVisualization()
        if (self.control_opponent):
            self.print_ok('Controller will control opponent')
        if (self.adaptive_Qop):
            self.print_ok('Qop1 = %.2f/%.2f, Qop2 = %.2f'%(self.Qop1_following,self.Qop1_leading, self.Qop2))
        else:
            self.print_ok('Qop1 = %.2f, Qop2 = %.2f'%(self.Qop1, self.Qop2))
        if (self.blocking_control):
            self.print_ok(f'blocking control enabled Qop {self.Qop1_blocking}')


    def init(self):
        if (self.linearize_around_zero_control):
            self.print_warning('----- Linearizing around u=0 ----- ')
        self.simulator = self.main.simulator
        assert(isinstance(self.simulator,CurvilinearSimulator))
        assert(len(self.main.cars)==2)
        self.ego_car = self.car
        for car in self.main.cars:
            if car != self.ego_car:
                self.oppo_car = car
                break

        delta_x = self.ego_car.sim_states - self.oppo_car.sim_states
        self.start_lead_i_j = delta_x[0]


    def final(self):
        delta_x = self.ego_car.sim_states - self.oppo_car.sim_states
        self.end_lead_i_j = delta_x[0]
        self.print_info(f'ego car : {self.ego_car.sim_states}')
        self.print_info(f'opponent car : {self.oppo_car.sim_states}')
        self.t.summary()
        self.lqt.summary()

    def isInCollision(self):
        delta_x = self.ego_car.sim_states - self.oppo_car.sim_states
        is_in_collision = np.abs(delta_x[0])<self.opponent_min_distance_s and np.abs(delta_x[2])<self.opponent_min_distance_n
        return is_in_collision

    def control(self):
        self.debug_dict = {}
        # s,v,n,phi
        ctrl0, ctrl1 = self.lqControl(self.ego_car.sim_states, self.oppo_car.sim_states)

        # car i
        car_i = self.ego_car
        car_j = self.oppo_car

        #enforce control constraint
        bounded_ctrl,constrained = self.boundControl(ctrl0,car_i)
        car_i.steering = bounded_ctrl[0]
        car_i.throttle = bounded_ctrl[1]
        if (self.debug):
            car0_coord = car_i.states[0:2]
            car0_heading = car_i.states[2]
            left, right = self.main.track.preciseTrackBoundary(car0_coord, car0_heading)
            ctrl0_normalized = np.linalg.norm([ctrl0[0]/car_i.max_ay, ctrl0[1]/car_i.max_ax])
            ctrl0_text = f'car0 red: v = {car_i.states[3]:.2f} S: {car_i.steering:.2f} T: {car_i.throttle:.2f}'
            if (left<0 or right<0):
                self.print_warning(ctrl0_text+' ---- out of track ')
            else:
                if (constrained):
                    self.print_ok(ctrl0_text+' C')
                else:
                    self.print_info(ctrl0_text)

        # car j
        if (self.control_opponent):
            bounded_ctrl,constrained = self.boundControl(ctrl1,car_j)
            car_j.steering = bounded_ctrl[0]
            car_j.throttle = bounded_ctrl[1]

            if (self.debug):
                car1_coord = car_j.states[0:2]
                car1_heading = car_j.states[2]
                left, right = self.main.track.preciseTrackBoundary(car1_coord, car1_heading)
                ctrl1_normalized = np.linalg.norm([ctrl1[0]/car_j.max_ay, ctrl1[1]/car_j.max_ax])
                ctrl1_text = f'car1 gre: v = {car_j.states[3]:.2f} S: {car_j.steering:.2f} T: {car_j.throttle:.2f}'
                if (left<0 or right<0):
                    self.print_warning(ctrl1_text+' ---- out of track ')
                else:
                    if (constrained):
                        self.print_ok(ctrl1_text+' C')
                    else:
                        self.print_info(ctrl1_text)
        if (self.draw_prediction):
            self.drawPredictedTrajectory()
        #self.drawDebug()
        return

    def boundControl(self, control, car):
        violated = False
        v = car.states[3]
        max_acc = car.max_ax * (1-v/car.max_v)
        # first scale to ellipse y/aym^2+x/axm^2=1
        # then cap ax to  (-infty,max_acc]
        ay_normalized = control[0]/car.max_ay
        ax_normalized = control[1]/car.max_ax
        theta = np.arctan2(ax_normalized,ay_normalized)
        r = np.linalg.norm([ax_normalized,ay_normalized])
        if (r>1.0):
            r = 1.0
            violated = True
        ay = car.max_ay*r*np.cos(theta)
        ax = car.max_ax*r*np.sin(theta)
        if (ax > max_acc):
            ax = max_acc
            violated = True
        return (ay,ax),violated



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
        #l_op(x,xop) = (x-xop)T Qcol (x-xop) = (remove const) xT Qcol x - 2xopT Qcol x
        #l_path = sym.xQx_diag(sym.x,self.Q) + sym.product(self.q, sym.x) + self.xQx_diag(sym.u, self.R)
        #l_op = sym.xQx_diag(sym.minus(sym.x,sym.xop), self.Qcol)
        #sym.l = l_path + l_op
        sym.symDer()
        return sym


    def linearizeSymbolic(self,x0,u0):
        ''' linearize dynamics symbolically '''
        sym = self.sym
        x0 = x0.flatten()
        u0 = u0.flatten()
        #xop = xop.flatten()
        k_s = self.simulator.curvature(x0[0])
        subs_dict = {sym.k_s:k_s}
        '''
        for i in range(self.n):
            subs_dict.update({sym.xop[i]:xop[i]})
        '''

        #fx,fu,lx,lu,lxx,luu,lux = self.sym.calcDer(x0=x0, u0=u0, subs_dict=subs_dict)
        fx,fu = self.sym.calcDer(x0=x0, u0=u0, subs_dict=subs_dict)
        return fx,fu

    def linearizeManual(self,x,u):
        ''' linearize manually using equations from sympy'''
        x0,x1,x2,x3 = x.flatten()
        u0,u1 = u.flatten()
        k_s = self.simulator.curvature(x0)

        dfdx = [[1, 0.01*cos(x3)/(-k_s*x2 + 1), 0.01*k_s*x1*cos(x3)/(-k_s*x2 + 1)**2, -0.01*x1*sin(x3)/(-k_s*x2 + 1)], [0, 1, 0, 0], [0, 0.01*sin(x3), 1, 0.01*x1*cos(x3)], [0, -0.01*k_s*cos(x3)/(-k_s*x2 + 1) - 0.01*u0/x1**2, -0.01*k_s**2*x1*cos(x3)/(-k_s*x2 + 1)**2, 0.01*k_s*x1*sin(x3)/(-k_s*x2 + 1) + 1]]

        dfdu = [[0, 0], [0, 0.0100000000000000], [0, 0], [0.01/x1, 0]]


        return np.array(dfdx,dtype=np.float64),np.array(dfdu,dtype=np.float64)

    def update_dynamics(self,states,controls,dt=None):
        if (dt is None):
            dt = self.dt
        return self.simulator.advancePointMassDynamics(states.flatten(),controls.flatten(),dt)

    def lqControl(self,x0_i,x0_j):
        self.t.s()
        P1s = np.array([np.zeros((self.m,self.n*2))]*self.horizon)
        P2s = np.array([np.zeros((self.m,self.n*2))]*self.horizon)
        alpha1s = np.array([np.zeros((self.m,1))]*self.horizon)
        alpha2s = np.array([np.zeros((self.m,1))]*self.horizon)
        if (self.linearize_around_zero_control):
            self.u_i_ref = np.zeros((self.horizon, self.m,1))
            self.u_j_ref = np.zeros((self.horizon, self.m,1))
        # FIXME
        self.u_i_ref = np.zeros((self.horizon, self.m,1))
        self.u_j_ref = np.zeros((self.horizon, self.m,1))
        car_i = self.ego_car
        car_j = self.oppo_car


        # iterations
        for iteration in range(self.iterations):
            # roll out u_ref, get x_ref
            # linearize around _ref, get A,B,d
            xx_i =[x0_i.reshape((self.n,1))]
            Ais = []
            Bis = []
            uu_i = []

            xx_j =[x0_j.reshape((self.n,1))]
            Ajs = []
            Bjs = []
            uu_j = []

            # u_ki = u_ref_ki - P_ki @ dx_k - alpha_ki
            for t in range(self.horizon):
                #for ego agent i
                # x+ = A x + B u + d, for x~x_ref
                # ~x = x - x_ref
                # ~x+ = A~x + B~u
                dx_i = xx_i[-1] - self.x_i_ref[t]
                dx_j = xx_j[-1] - self.x_j_ref[t]
                dx = np.vstack([dx_i,dx_j])

                # NOTE ignoring control constraint
                #for ego agent i
                u = self.u_i_ref[t] - P1s[t] @ dx + alpha1s[t]
                #u,constrained = self.boundControl(u.flatten(),car_i)
                u = np.array(u).reshape(-1,1)
                self.t.s('update_dynamics')
                new_x = self.update_dynamics(xx_i[-1],u)
                self.t.e('update_dynamics')
                self.t.s('linearize')
                if (self.linearize_around_zero_control):
                    A, B = self.linearizeManual(xx_i[-1],np.zeros(self.m))
                else:
                    A, B = self.linearizeManual(xx_i[-1],u)
                self.t.e('linearize')
                xx_i.append(new_x.reshape(4,1))
                Ais.append(A)
                Bis.append(B)
                uu_i.append(u)

                #for ego agent j
                u = self.u_j_ref[t] - P2s[t] @ dx + alpha2s[t]
                # NOTE ignoring control constraint
                #u,constrained = self.boundControl(u.flatten(),car_j)
                u = np.array(u).reshape(-1,1)
                self.t.s('update_dynamics')
                new_x = self.update_dynamics(xx_j[-1],u)
                self.t.e('update_dynamics')
                self.t.s('linearize')
                if (self.linearize_around_zero_control):
                    #A,B,d = self.linearizeNumerical(xx_j[-1],np.zeros(self.m))
                    #A, B = self.linearizeSymbolic(xx_j[-1],np.zeros(self.m))
                    A, B = self.linearizeManual(xx_j[-1],np.zeros(self.m))
                else:
                    #A,B,d = self.linearizeNumerical(xx_j[-1],u)
                    #A, B = self.linearizeSymbolic(xx_j[-1],u)
                    A, B = self.linearizeManual(xx_j[-1],u)
                self.t.e('linearize')
                xx_j.append(new_x.reshape(4,1))
                Ajs.append(A)
                Bjs.append(B)
                uu_j.append(u)

            self.x_i_ref = xx_i
            self.u_i_ref = uu_i
            self.x_j_ref = xx_j
            self.u_j_ref = uu_j
            n = self.n
            m = self.m

            As = [block_diag(Ais[i],Ajs[i]) for i in range(len(Ais))]
            ds = [np.zeros(n*2) for i in range(len(Ais))]
            B1s = [np.vstack([Bi,np.zeros((n,m))]) for Bi in Bis]
            B2s = [np.vstack([np.zeros((n,m)),Bj]) for Bj in Bjs]


            self.t.s('getCostMatrices')
            Q1s,q1s,Q2s,q2s,Rs,rs = self.getCostMatrices(xx_i,uu_i,xx_j,uu_j)
            self.t.e('getCostMatrices')
            if (self.main.breakpoint.isSet()):
                breakpoint()
                self.main.breakpoint.clear()

            '''
            self.t.s('solve_lq_game')
            # LQ cost function, get Q,l, Rs
            [P1s_old, P2s_old], [alpha1s_old, alpha2s_old] = solve_lq_game(
                As, [B1s, B2s],
                [Q1s, Q2s], [q1s, q2s], Rs)
            alpha1s_old *= -1
            alpha2s_old *= -1
            self.t.e('solve_lq_game')
            '''

            self.t.s('my_solve_lq_game')
            # LQ cost function, get Q,l, Rs
            [new_P1s, new_P2s], [new_alpha1s, new_alpha2s] = my_solve_lq_game(
                As, [B1s, B2s],
                [Q1s, Q2s], [q1s, q2s], [Rs[0][0], Rs[1][1]],rs,ds,self.lqt)

            self.t.e('my_solve_lq_game')


            if (iteration == 0):
                alpha1s = new_alpha1s
                alpha2s = new_alpha2s
                P1s = new_P1s
                P2s = new_P2s
            else:
                alpha = self.alpha
                alpha1s = alpha1s* (1-alpha) + alpha *new_alpha1s
                alpha2s = alpha2s* (1-alpha) + alpha *new_alpha2s
                P1s = P1s* (1-alpha) + alpha *new_P1s
                P2s = P2s* (1-alpha) + alpha *new_P2s

        # additional layer of optimization
        if (self.blocking_control):
            K = len(Ais)
            # prefix 'b' signal blocking, to distinguish from As, Bs
            bAs = [As[k] - B2s[k] @ P2s[k] for k in range(K)]
            bBs = B1s
            bds = [ (B2s[k] @ alpha2s[k]).flatten() for k in range(K)]

            original_Qop = self.Qop1
            self.Qop1 = self.Qop1_blocking
            bQ1s,bq1s,_,_,bRs,brs = self.getCostMatrices(xx_i,uu_i,xx_j,uu_j)

            [blocking_P1s], [blocking_alpha1s] = my_solve_lq_game(
                bAs, [bBs],
                [bQ1s], [bq1s], [Rs[0][0]],[rs[0]],bds,self.lqt)

            self.Qop1 = original_Qop
            #self.print_info(np.linalg.norm(blocking_P1s-new_P1s))
            #self.print_info(np.linalg.norm(blocking_alpha1s-new_alpha1s))
            alpha1s = new_alpha1s = blocking_alpha1s
            P1s = new_P1s = blocking_P1s


        dx_i = xx_i[0] - self.x_i_ref[0]
        dx_j = xx_j[0] - self.x_j_ref[0]
        dx = np.vstack([dx_i,dx_j])
        ctrl1 = (self.u_i_ref[0] - P1s[0] @ dx + alpha1s[0]).flatten()
        ctrl2 = (self.u_j_ref[0] - P2s[0] @ dx + alpha2s[0]).flatten()

        self.debug_dict.update({'u_ref':np.array(self.u_i_ref), 'x_ref':np.array(self.x_i_ref), 'x_j_ref':np.array(self.x_j_ref), 'u_j_ref':np.array(self.u_j_ref)})
        self.t.e()
        return ctrl1,ctrl2

    def createConstants(self):
        n = self.n
        m = self.m
        self.II = np.hstack([np.eye(n),-np.eye(n)])
        self.R0 = np.zeros((m,m))
        self.Q1_x =  block_diag(self.Q1,np.zeros((n,n)))
        self.q1_x = np.hstack([self.q1.T,np.zeros((1,n))])
        self.Q2_x =  block_diag(np.zeros((n,n)),self.Q2)
        self.q2_x = np.hstack([np.zeros((1,n)),self.q2.T])
        self.R1 = 0.005*np.diag([1.0,1.0])
        self.R2 = 0.005*np.diag([1.0,1.0])

    def getCostMatrices(self,xx_i,uu_i,xx_j,uu_j):
        Q1s = []
        Q2s = []
        q1s = []
        q2s = []

        R11s = []
        R22s = []

        R12s = []
        R21s = []

        r1s = []
        r2s = []

        n = self.n
        m = self.m
        #II = np.hstack([np.eye(n),-np.eye(n)])
        #R0 = np.zeros((m,m))
        II = self.II
        R0 = self.R0

        for t in range(self.horizon):
            delta_x = xx_i[t] - xx_j[t]
            # these cost matrices work on the stacked agent state x, not state perturbation dx
            # cost_i = 1/2 x.T @ Qi_x @ x + qi_x.T @ x + 1/2 ui.T @ R @ ui + ri.T @ ui
            #Q1_x =  block_diag(self.Q1,np.zeros((n,n)))
            #q1_x = np.hstack([self.q1.T,np.zeros((1,n))])
            Q1_x =  self.Q1_x.copy()
            q1_x = self.q1_x.copy()
            if (self.adaptive_Qop):
                if (delta_x[0]>0):
                    q1_x[0,n] = self.Qop1_leading
                else:
                    q1_x[0,n] = self.Qop1_following

            else:
                q1_x[0,n] = self.Qop1

            #Q2_x =  block_diag(np.zeros((n,n)),self.Q2)
            #q2_x = np.hstack([np.zeros((1,n)),self.q2.T])
            Q2_x =  self.Q2_x.copy()
            q2_x = self.q2_x.copy()
            q2_x[0,0] = self.Qop2

            # barrier function: opponent collision
            if (np.abs(delta_x[0])<self.opponent_min_distance_s and np.abs(delta_x[2])<self.opponent_min_distance_n):
                #self.print_info('collision avoidance')
                sgn_s = -1 if delta_x[0]>0 else 1
                sgn_n = -1 if delta_x[2]>0 else 1
                # based on current position, agent in front ignorant of collision
                if (self.leader_collision_ignorant and np.abs(xx_i[t][0]-xx_j[t][0]) > self.opponent_min_distance_s):
                    if (xx_i[t][0] - xx_j[t][0] > 0):
                        Q2_x += 2* II.T @ self.Qcol @ II
                        q2_x += (np.array([[sgn_s*2*self.opponent_min_distance_s, 0, sgn_n*2*self.opponent_min_distance_n, 0]]) @ self.Qcol @ II)
                    else:
                        Q1_x += 2* II.T @ self.Qcol @ II
                        q1_x += (np.array([[sgn_s*2*self.opponent_min_distance_s, 0, sgn_n*2*self.opponent_min_distance_n, 0]]) @ self.Qcol @ II)
                else:
                    # if side by side both agent responsible
                    Q1_x += 2* II.T @ self.Qcol @ II
                    q1_x += (np.array([[sgn_s*2*self.opponent_min_distance_s, 0, sgn_n*2*self.opponent_min_distance_n, 0]]) @ self.Qcol @ II)
                    Q2_x += 2* II.T @ self.Qcol @ II
                    q2_x += (np.array([[sgn_s*2*self.opponent_min_distance_s, 0, sgn_n*2*self.opponent_min_distance_n, 0]]) @ self.Qcol @ II)

            # barrier function: track boundary
            cart_states_i = self.simulator.curv2Cart(xx_i[t].flatten())
            self.t.s('preciseTrackBoundary')
            left_boundary_i, right_boundary_i = self.main.track.preciseTrackBoundary(cart_states_i[:2],cart_states_i[2])
            #left_boundary_i, right_boundary_i = self.main.track.preciseTrackBoundary(s=xx_i[t][0],n=xx_i[t][2])
            self.t.e('preciseTrackBoundary')

            # n>0 -> left
            if (left_boundary_i < self.boundary_min_distance):
                Q1_x += 2* np.diag([0,0,self.boundary_cost,0,0,0,0,0])
                q1_x += 2* np.array([[0,0,-self.boundary_cost*2*self.boundary_min_distance,0,0,0,0,0]])
            elif (right_boundary_i < self.boundary_min_distance):
                Q1_x += 2* np.diag([0,0,self.boundary_cost,0,0,0,0,0])
                q1_x += 2* np.array([[0,0,+self.boundary_cost*2*self.boundary_min_distance,0,0,0,0,0]])

            cart_states_j = self.simulator.curv2Cart(xx_j[t].flatten())
            self.t.s('preciseTrackBoundary')
            left_boundary_j, right_boundary_j = self.main.track.preciseTrackBoundary(cart_states_j[:2],cart_states_j[2])
            #left_boundary_j, right_boundary_j = self.main.track.preciseTrackBoundary(s=xx_j[t][0],n=xx_j[t][2])
            self.t.e('preciseTrackBoundary')

            # n>0 -> left
            if (left_boundary_j < self.boundary_min_distance):
                Q2_x += 2* np.diag([0,0,0,0,0,0,self.boundary_cost,0])
                q2_x += 2* np.array([[0,0,0,0,0,0,-self.boundary_cost*2*self.boundary_min_distance,0]])
            elif (right_boundary_j < self.boundary_min_distance):
                Q2_x += 2* np.diag([0,0,0,0,0,0,self.boundary_cost,0])
                q2_x += 2* np.array([[0,0,0,0,0,0,+self.boundary_cost*2*self.boundary_min_distance,0]])


            # barrier function: control limit
            # with ax, ay being a control this is more difficult
            car_i = self.ego_car
            car_j = self.oppo_car
            # cost on control u (ay,ax)
            # normal ctrl cost: (ay/ay_max-1)**2 + (ax/ax_max-1)**2
            #R1 = 0.01*np.diag([1.0/car_i.max_ay,1.0/car_i.max_ax])
            #R2 = 0.01*np.diag([1.0/car_j.max_ay,1.0/car_j.max_ax])
            #R1 = 0.005*np.diag([1.0,1.0])
            #R2 = 0.005*np.diag([1.0,1.0])
            R1 = self.R1
            R2 = self.R2
            r1_x = np.zeros((1,self.m))
            r2_x = np.zeros((1,self.m))

            if (self.circular_control_barrier):
                # normalized ay,ax for agent i
                bounded_ctrl,constrained = self.boundControl(uu_i[t].flatten(),car_i)
                if ( constrained ):
                    #self.print_info('car 0 control barrier')
                    # the point on traction circle that's closest to current (ay,ax)
                    by = bounded_ctrl[0]
                    bx = bounded_ctrl[1]
                    R1 = self.control_barrier_cost * np.diag([2, 2])
                    r1_x = self.control_barrier_cost * np.array([[-2*by, -2*bx]])

                # normalized ay,ax for agent j
                bounded_ctrl,constrained = self.boundControl(uu_j[t].flatten(),car_j)
                if ( constrained ):
                    by = bounded_ctrl[0]
                    bx = bounded_ctrl[1]
                    #self.print_info('car 1 control barrier')
                    R2 = self.control_barrier_cost * np.diag([2, 2])
                    r2_x = self.control_barrier_cost * np.array([[-2*by, -2*bx]])
            else:
                self.print_error('this has shown to be uneffective')
                # linear control barrier
                ayi_n = uu_i[t][0]/car_i.max_ay
                axi_n = uu_i[t][1]/car_i.max_ax
                if ( (axi_n)**2 + (ayi_n)**2 > 1.0):
                    ax = uu_i[t][1].item()
                    ay = uu_i[t][0].item()
                    axm = car_i.max_ax
                    aym = car_i.max_ay
                    theta = np.arctan2(ax/axm, ay/aym)
                    p = [aym*np.cos(theta), axm*np.sin(theta)]
                    C = -(p[1] * ax/axm**2 + p[0]*ay/aym**2)
                    R1 = self.control_barrier_cost*2*np.array([[ay**2/aym**4,ax*ay/(axm**2*aym**2)], [ax*ay/(axm**2*aym**2), ax**2/axm**4]])
                    r1_x = self.control_barrier_cost * C*np.array([[2*ay/aym**2, 2*ax/axm**2]])

                # normalized ay,ax for agent i
                ayj_n = uu_j[t][0]/car_j.max_ay
                axj_n = uu_j[t][1]/car_j.max_ax
                if ( (uu_j[t][1]/car_j.max_ax)**2 + (uu_j[t][0]/car_j.max_ay)**2 > 1.0):
                    ax = uu_j[t][1].item()
                    ay = uu_j[t][0].item()
                    axm = car_j.max_ax
                    aym = car_j.max_ay
                    theta = np.arctan2(ax/axm, ay/aym)
                    p = [aym*np.cos(theta), axm*np.sin(theta)]
                    C = -(p[1] * ax/axm**2 + p[0]*ay/aym**2)
                    R2 = self.control_barrier_cost*2*np.array([[ay**2/aym**4,ax*ay/(axm**2*aym**2)], [ax*ay/(axm**2*aym**2), ax**2/axm**4]])
                    r2_x = self.control_barrier_cost * C*np.array([[2*ay/aym**2, 2*ax/axm**2]])

            xx_ref = np.vstack([xx_i[t], xx_j[t]])
            # these work on state perturbation dx
            Q1 = Q1_x
            q1 = xx_ref.T @ Q1_x + q1_x

            Q2 = Q2_x
            q2 = xx_ref.T @ Q2_x + q2_x

            Q1s.append(Q1)
            Q2s.append(Q2)
            q1s.append(q1.T)
            q2s.append(q2.T)

            R11s.append(R1)
            R22s.append(R2)

            R12s.append(R0)
            R21s.append(R0)

            r1 = ( uu_i[t].T@ R1 + r1_x).T
            r2 = ( uu_j[t].T@ R2 + r2_x).T

            r1s.append( r1 )
            r2s.append( r2 )

        Rs = [[R11s, R12s], [R21s, R22s]]
        rs = [r1s, r2s]

        return Q1s,q1s,Q2s,q2s,Rs,rs


    # differentiate dynamics around nominal state and control
    # return: A, B, d, s.t. x_k+1 = Ax + Bu + d
    def linearizeNumerical(self, nominal_state, nominal_ctrl):
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

    def drawDebug(self):
        if (self.main.visualization.update_visualization.is_set()):
            img = self.main.visualization.visualization_img
            car0 = self.ego_car
            car0_coord = car0.states[0:2]
            car0_heading = car0.states[2]
            left, right = self.main.track.preciseTrackBoundary(car0_coord, car0_heading)
            left_pt = [car0_coord[0] + np.cos(car0_heading+np.pi/2)*left, car0_coord[1] + np.sin(car0_heading+np.pi/2)*left]
            right_pt = [car0_coord[0] - np.cos(car0_heading+np.pi/2)*right, car0_coord[1] - np.sin(car0_heading+np.pi/2)*right]
            img = self.main.track.drawPolyline([left_pt,right_pt],img=img)

            self.main.visualization.visualization_img = img

    def drawPredictedTrajectory(self, lineColor=(0,100,100)):
        self.drawTrajectory(self.x_i_ref, lineColor=(0,100,100))
        if (self.control_opponent):
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

    def overrideControlVisualization(self):
        # override control visualization from Visualization.py
        # self.main.visualization.drawControlStaticForAllCars = lambda img: img

        def drawControl(img,car,coord):
            ctrl = np.array((car.steering, car.throttle))
            # get the control limit, in the direction of current control
            ctrl_limit,constrained = self.boundControl(ctrl*1000,car)
            ctrl_limit = np.array(ctrl_limit)
            if (not constrained):
                ctrl_limit[0] = car.max_ay
                ctrl_limit[1] = car.max_ax
            else:
                ctrl_limit[0] = np.abs(ctrl_limit[0])
                ctrl_limit[1] = np.abs(ctrl_limit[1])

            def bound(a,l,h):
                val = l if a < l else a
                return h if val>h else val
            def map(val, in_l, in_h, out_low, out_high):
                # out of bound flag
                oob = False
                if (val<in_l):
                    #val = in_l
                    oob = True
                elif (val > in_h):
                    #val = in_h
                    oob = True
                val = (val-in_l)/(in_h-in_l)*(out_high-out_low)+out_low
                if (isnan(val)):
                    val = 0.0
                return val, oob

            #x1 and y1 are the origin values -- need to be changed if origin changes
            x1 = coord[0] + 30
            y1 = coord[1]
            x,y,heading, vf_lf, vs_lf, omega_lf = car.states
            # Add steering bar
            steering,oob = map(car.steering, -ctrl_limit[0], ctrl_limit[0], 100,0)
            img = cv2.rectangle(img, (x1 , y1 + 25), (x1 + 100, y1 + 40), (0, 0, 255), 1)
            if (oob):
                img = cv2.rectangle(img, (x1 + 50, y1 + 25), (x1 + int(steering), y1 + 40), (0, 0, 255), -1)
            else:
                img = cv2.rectangle(img, (x1 + 50, y1 + 25), (x1 + int(steering), y1 + 40), (0, 255, 0), -1)
            img = cv2.putText(img, 'Steering', (x1 + 104, y1 + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            # Add Throttle bar
            throttle,oob = map(car.throttle, -ctrl_limit[1], ctrl_limit[1], 0,100)
            img = cv2.rectangle(img, (x1 , y1 + 45), (x1 + 100, y1 + 60), (0,0,255), 1)
            if (oob):
                img = cv2.rectangle(img, (x1 + 52, y1 + 45), (x1 + int(throttle), y1 + 60), (0, 0, 255), -1)
            else:
                img = cv2.rectangle(img, (x1 + 52, y1 + 45), (x1 + int(throttle), y1 + 60), (0, 255, 0), -1)
            img = cv2.putText(img, 'Throttle', (x1 + 104, y1 + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            if (self.isInCollision()):
                img = cv2.putText(img, 'Collision', (x1 + 50, y1 + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            return img

        self.main.visualization.drawControl = drawControl
