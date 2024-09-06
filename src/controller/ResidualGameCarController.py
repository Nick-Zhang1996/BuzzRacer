from common import *
from math import isnan,pi,degrees,radians,sin,cos
from controller.CarController import CarController
from controller.PidController import PidController
from third_party.solve_lq_game import solve_lq_game
from extension.simulator.CurvilinearSimulator import CurvilinearSimulator
from util.SymbolicDynamics import SymbolicDynamics
from scipy.linalg import block_diag
import sympy
from util.TimeUtil import TimeUtil
import cv2

# FIXME dummy placeholder to check access
#import os
#import sys
#sys.path.append(os.path.abspath('../../hybrid'))
#from src.build.car_merge_kinematic_bicycle import CarMergeKinematicBicycle as cpp_CarMergeKinematicBicycle
from controller.RD3G.CarRacing import CarRacing

class ResidualGameCarController(CarController):
    def __init__(self, car,config):
        super().__init__(car,config)
        self.t = TimeUtil(True)
        self.lqt = TimeUtil(True)
        self.m = 2
        self.n = 4
        self.iterations = 3
        self.draw_prediction = True
        # aggressiveness
        self.alpha = 0

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

        self.residual_game = CarRacing()

        # if true, this controller will control opponent
        self.control_opponent = False
        ConfigObject.__init__(self,config)

    def preInit(self):
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

    def solveGame(self, x0, x1):
        # TODO start here

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
        ctrl0, ctrl1 = self.residualGameControl(self.ego_car.sim_states, self.oppo_car.sim_states)

        # car i
        car_i = self.ego_car
        car_j = self.oppo_car

        #enforce control constraint
        bounded_ctrl,constrained = self.boundControl(ctrl0,car_i)
        car_i.steering = bounded_ctrl[0]
        car_i.throttle = bounded_ctrl[1]

        # car j
        if (self.control_opponent):
            bounded_ctrl,constrained = self.boundControl(ctrl1,car_j)
            car_j.steering = bounded_ctrl[0]
            car_j.throttle = bounded_ctrl[1]

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

    def residualGameControl(self,x0_i,x0_j):
        self.t.s()
        #self.residual_game.x0 = np.
        self.t.e()
        ctrl1 = ctrl2 = np.zeros(self.m)
        return ctrl1,ctrl2

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

