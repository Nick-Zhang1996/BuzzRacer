# MPC controller for dynamic bicycle model
# used by car.py
import numpy as np
from mpc import MPC
from time import time
# from timeUtil import ExecutionTimer
from buzzracer.cars import car
from math import atan2, radians, degrees, sin, cos, pi, tan, copysign, asin, acos, isnan, exp, pi
from buzzracer.common import *

import os
import sys
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), './mppi/')
sys.path.append(base_dir)


class ctrlMpcWrapper(car):
    def __init__(self, car_setting, dt):
        super().__init__(car_setting, dt)
        # self.t = ExecutionTimer(True)
        self.freq = []
        self.min_freq = 999
        self.dt = dt
        self.prediction_steps = 30

        g = 9.81
        self.m = 0.1667
        self.Caf = 5*0.25*self.m*g
        # self.Car = 5*0.25*self.m*g
        self.Car = self.Caf
        # longitudinal speed
        self.Vx = 1.0
        self.Vy = 0
        # CG to front axle
        self.lf = 0.09-0.036
        self.lr = 0.036
        # approximate as a solid box
        self.Iz = self.m/12.0*(0.1**2+0.1**2)
        return

# given state of the vehicle and an instance of track, provide throttle and steering output
# input:
#   state: (x,y,heading,v_forward,v_sideway,omega)
#   track: track object, can be RCPtrack or skidpad
#   v_override: If specified, use this as target velocity instead of the optimal value provided by track object
#   reverse: true if running in opposite direction of raceline init direction

# output:
#   (throttle,steering,valid,debug)
# ranges for output:
#   throttle -1.0,self.max_throttle
#   steering as an angle in radians, TRIMMED to self.max_steering, left(+), right(-)
#   valid: bool, if the car can be controlled here, if this is false, then throttle will also be set to 0
#           This typically happens when vehicle is off track, and track object cannot find a reasonable local raceline
# debug: a dictionary of objects to be debugged, e.g. {offset, error in v}
    def ctrl_car(self, state, track, v_override=None, reverse=False):
        # state dimension
        n = 5
        # action dimension
        m = 1
        # output dimension(y=Cx)
        l = 1

        tic = time()
        # t = self.t
        t.s()

        p = self.prediction_steps
        dt = self.dt

        debug_dict = {}
        t.s('get ref point')
        e_cross, e_heading, v_ref, k_ref, coord_ref, valid = track.get_ref_point(
            state, p, dt, reverse=reverse)
        debug_dict['crosstrack_error'] = e_cross
        debug_dict['heading_error'] = e_heading
        t.e('get ref point')

        t.s('assemble ref')
        if not valid:
            ret = (0, 0, False, debug_dict)
            return ret

        # first element of _ref is current state, we don't need that
        # FIXME
        v_target = v_ref[0]
        # v_target *= 0.5
        v_ref = v_ref[1:]
        k_ref = k_ref[1:]
        # only reference we need is dpsi_dt = Vx * K
        dpsi_dt_ref = v_ref * k_ref

        # x,y,heading,vf,vs,omega = state
        # vx = vf * cos(heading) - vs*sin(heading)
        # vy = vf * sin(heading) + vs*cos(heading)

        # assemble x0
        # state var for dynamic bicycle model w.r.t. lateral and heading error
        # e_lateral, dedt_lateral, e_heading,dedt_heading, 1(unity)
        x0 = np.array([e_cross, 0, e_heading, 0, 1])

        x, y, heading, vf, vs, omega = state
        t.e('assemble ref')

        t.s('assemble matrix')
        # assemble Ak matrices
        Car = self.Car
        Caf = self.Caf
        mass = self.m
        lf = self.lf
        lr = self.lr
        Iz = self.Iz
        Vx = vf

        def getA_raw(Vx, dpsi_r): return \
            np.array([[0, 1, 0, 0, 0],
                      [0, -(2*Caf+2*Car)/mass/Vx, (2*Caf+2*Car)/mass, (-2*Caf*lf+2 *
                                                                       Car*lr)/mass/Vx, (-(2*Caf*lf-2*Car*lr)/mass/Vx - Vx)*dpsi_r],
                      [0, 0, 0, 1, 0],
                      [0, -(2*Caf*lf-2*Car*lr)/Iz/Vx, (2*Caf*lf-2*Car*lr)/Iz, -(2*Caf *
                                                                                lf*lf+2*Car*lr*lr)/Iz/Vx, -(2*Caf*lf*lf+2*Car*lr*lr)/Iz/Vx*dpsi_r],
                      [0, 0, 0, 0, 0]])

        # assemble Bk matrices
        B = np.array([[0, 2*Caf/mass, 0, 2*Caf*lf/Iz, 0]]).T

        # since
        # self.state = self.state + (Ak @ self.state + Bk @ u)*dt
        # we now compute augmented A and B
        In = np.eye(n)
        def get_a(Vx, dpsi_r): return In + getA_raw(Vx, dpsi_r) * dt

        A_vec = [get_a(Vx, dpsi_r) for Vx, dpsi_r in zip(v_ref, dpsi_dt_ref)]
        # LTI model
        # v0 = v_ref[0]
        # A_vec = [get_a(Vx,dpsi_r) for Vx,dpsi_r in zip([v0]*len(v_ref),[0]*len(v_ref))]

        B_vec = [B*dt] * p

        # define output matrix C, for a single state vector
        # y = Cx
        C = np.zeros([l, n])
        C[0, 0] = 1

        # J = y.T P y + u.T Q u
        P = np.zeros([l, l])
        # e_cross
        P[0, 0] = 1

        Q = np.zeros([m, m])
        Q[0, 0] = 1e-3
        y_ref = np.zeros([p, l])
        x0 = x0
        p = p

        # 5 deg/s
        # typical servo speed 60deg/0.1s
        # u: steering
        du_max = np.array([radians(60)/0.1*dt])*0.5
        u_max = np.array([radians(25)])
        t.e('assemble matrix')

        t.s('convert problem')
        self.mpc.convert_ltv(A_vec, B_vec, C, P, Q, y_ref, x0, du_max, u_max)
        t.e('convert problem')

        t.s('solve')
        u_optimal = self.mpc.solve()
        t.e('solve')

        t.s('actuate')
        # u is stacked, so [throttle_0,steering_0, throttle_1, steering_1]
        # plt.plot(u_optimal[1::2,0])
        # plt.show()
        steering = u_optimal[0]
        # print(degrees(steering))

        # throttle is controller by other controller
        # throttle = u_optimal[0,1]
        throttle = self.calc_throttle(state, v_target)

        debug_dict['x_ref'] = coord_ref
        # debug_dict['x_ref'] = []
        # debug_dict['x_project'] = self.mpc.debug()
        ret = (throttle, steering, True, debug_dict)
        t.e('actuate')
        tac = time()
        self.freq.append(tac-tic)
        if len(self.freq) > 300:
            self.freq.pop(0)
        # print("freq = %.2f"%(1.0/(tac-tic)))
        # print("mean freq = %.2f"%(1.0/(np.mean(self.freq))))
        t.e()
        if (1.0/(tac-tic) < self.min_freq):
            self.min_freq = 1.0/(tac-tic)
        # print("min freq = %.2f"%(self.min_freq))

        return ret

    # initialize mpc
    # sim: an instance of advCarSim so we have access to parameters
    def init_mpc_sim(self, sim):
        # prediction step
        self.prediction_steps = 15
        # prediction discretization dt
        # NOTE we may be able to use a finer time step in x ref calculation, this can potentially increase accuracy
        self.dt = 0.03
        # together p*mpc_dt gives prediction horizon

        self.Caf = sim.Caf
        self.Car = sim.Car
        self.lf = sim.lf
        self.lr = sim.lr
        self.Iz = sim.Iz
        self.m = sim.m
        self.mpc = MPC()
        # state dimension
        n = 5
        # action dimension
        m = 1
        # output dimension(y=Cx)
        l = 1
        p = self.prediction_steps
        self.mpc.setup(n, m, l, p)
        return

    # initialize mpc
    def init_mpc_real(self):
        # prediction step
        self.prediction_steps = 5
        # prediction discretization dt
        # NOTE we may be able to use a finer time step in x ref calculation, this can potentially increase accuracy
        self.dt = 0.03
        # together p*mpc_dt gives prediction horizon

        g = 9.81
        self.m = 0.1667
        self.Caf = 5*0.25*self.m*g
        # self.Car = 5*0.25*self.m*g
        self.Car = self.Caf
        # CG to front axle
        self.lf = 0.09-0.036
        self.lr = 0.036
        # approximate as a solid box
        self.Iz = self.m/12.0*(0.1**2+0.1**2)

        self.mpc = MPC()
        # state dimension
        n = 5
        # action dimension
        m = 1
        # output dimension(y=Cx)
        l = 1
        p = self.prediction_steps
        self.mpc.setup(n, m, l, p)
        return

    # --- Deprecated ---
    # get future reference point for dynamic MPC
    # Inputs:
    # state: vehicle state, same as in self.local_trajectory()
    # p : lookahead steps
    # dt : time between each lookahead steps

    # Return:
    # xref : np array of size (p+1)*2,
    # there are p+1 entries because xref0 is the ref point for current location,
    # and then there are p projection points
    # psi_ref : reference heading at the reference points, size (p+1)*2
    # v_ref : reference heading at the reference points, size (p+1)*2
    # valid : a boolean indicating whether the function was able to find a valid result
    # The function first finds a point on trajectory closest to vehicle location
    # with local_trajectory(), then find p points down the trajectory that are spaced vk * dt
    # apart in path length. vk is the reference velocity at those points
    def get_ref_point(self, state, p, dt):
        t = self.t

        t.s()
        # set wheelbase to 0 to get point closest to vehicle CG
        t.s('local traj')
        retval = self.local_trajectory(
            state, wheelbase=0.102/2.0, return_u=True)
        t.e('local traj')
        if retval is None:
            return None, None, False

        # parse return value from local_trajectory
        (local_ctrl_pnt, offset, orientation, curvature, v_target, u0) = retval
        if isnan(orientation):
            return None, None, False

        # calculate s value for projection ref points
        t.s('find s')
        s0 = self.uToS(u0).item()
        v0 = self.targetVfromU(u0 % self.track_length_grid).item()
        der = splev(u0 % self.track_length_grid, self.raceline, der=1)
        heading0 = atan2(der[1], der[0])
        t.e('find s')

        t.s('curvature')

        def _norm(x):
            return np.linalg.norm(x, axis=0)
        # gives right sign for omega,
        # this is indep of track direction since it's calculated based off vehicle orientation

        dr = np.array(splev(u0 % self.track_length_grid, self.raceline, der=1))
        ddr = vec_curvature = np.array(
            splev(u0 % self.track_length_grid, self.raceline, der=2))
        cross_curvature = der[0]*vec_curvature[1]-der[1]*vec_curvature[0]
        curvature = 1.0/(_norm(dr)**3/(_norm(dr)**2*_norm(ddr)
                         ** 2 - np.sum(dr*ddr, axis=0)**2)**0.5)

        t.e('curvature')

        # curvature needs to be signed to indicate whether signage target angular velocity
        # a cross product gives right signage for omega,
        # this is indep of track direction since it's calculated based off vehicle orientation
        cross_curvature = der[0]*vec_curvature[1]-der[1]*vec_curvature[0]

        # k_vec.append(norm_curvature)
        # k_sign_vec.append(cross_curvature)
        k_vec = curvature
        k_sign_vec = cross_curvature

        s_vec = [s0]
        v_vec = [v0]
        heading_vec = [heading0]
        k_vec = [curvature]
        k_sign_vec = [cross_curvature]

        u_vec = [u0]

        t.s('main loop')
        for k in range(1, p+1):
            s_k = s_vec[-1] + v_vec[-1] * dt
            s_vec.append(s_k)
            # find ref velocity for projection ref points
            # TODO adjust ref velocity for current vehicle velocity
            # v_k = self.targetVfromU(u_k%self.track_length_grid)
            # v_k = self.sToV(s_k%self.raceline_len_m)
            v_k = self.sToV_lut(s_k % self.raceline_len_m)
            v_vec.append(v_k)
        t.e('main loop')

        # u_vec = np.array(u_vec)%self.track_length_grid
        # find ref heading for projection ref points
        t.s('psi')
        # der = np.array(splev(u_vec,self.raceline,der=1))
        s_vec = np.array(s_vec) % self.raceline_len_m
        der = np.array(splev(s_vec, self.raceline_s, der=1))
        # heading_k = atan2(der[1],der[0])
        # heading_vec.append(heading_k)
        t.e('psi')
        # find ref coordinates for projection ref points

        t.s('coord')
        coord_vec = np.array(splev(s_vec, self.raceline_s)).T
        t.e('coord')

        t.s('K')

        # norm_curvature = np.linalg.norm(vec_curvature,axis=1)
        dr = np.array(splev(s_vec, self.raceline_s, der=1))
        ddr = vec_curvature = np.array(splev(s_vec, self.raceline_s, der=2))

        curvature = 1.0/(_norm(dr)**3/(_norm(dr)**2*_norm(ddr)
                         ** 2 - np.sum(dr*ddr, axis=0)**2)**0.5)

        # curvature needs to be signed to indicate whether signage target angular velocity
        # a cross product gives right signage for omega,
        # this is indep of track direction since it's calculated based off vehicle orientation
        cross_curvature = der[0, :]*vec_curvature[1, :] - \
            der[1, :]*vec_curvature[0, :]

        # k_vec.append(norm_curvature)
        # k_sign_vec.append(cross_curvature)
        k_vec = curvature
        k_sign_vec = cross_curvature

        # TODO check dimension
        k_signed_vec = np.copysign(k_vec, k_sign_vec)

        x, y, heading, vf, vs, omega = state
        e_heading = ((heading - heading0) + pi/2.0) % (2*pi) - pi/2.0
        t.e('K')

        t.e()
        # return offset, e_heading, np.array(v_vec),np.array(k_signed_vec), np.array(coord_vec),True
        return offset, e_heading, np.array(v_vec), np.array(k_signed_vec), np.array(coord_vec), True
