from car import Car
from math import atan2,radians,degrees,sin,cos,pi,tan,copysign,asin,acos,isnan,exp,pi,atan
import numpy as np
from time import time,sleep
from timeUtil import execution_timer
from mppi import MPPI
from scipy.interpolate import splprep, splev,CubicSpline,interp1d
import matplotlib.pyplot as plt

from common import *

import os
import sys
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), './mppi/')
sys.path.append(base_dir)

class ctrlMppiWrapper(Car):
    def __init__(self,car_setting,dt):
        super().__init__(car_setting,dt)

        # given parameterized raceline x,y = R(s), this corresponds to raceline_s
        # last_s is the last s such that R(last_s) is closest to vehicle
        # used as a starting point for root finding
        self.last_s = None
        self.p = execution_timer(True)
        return

    # if running on real platform, set sim to None so that default values for car dimension/properties will be used
    def init(self,track,sim=None):
        self.track = track

        # NOTE NOTE NOTE
        # update mppi_racecar.cu whenever you change parameter here
        self.mppi_dt = 0.03
        self.samples_count = 1024*4
        self.discretized_raceline_len = 1024
        self.horizon_steps = 20
        self.control_dim = 2
        self.state_dim = 6
        self.temperature = 1.0
        # default
        self.noise_cov = np.diag([(self.max_throttle/2)**2,radians(40.0/2)**2])
        # restricting to test terminal cost
        #self.noise_cov = np.diag([(1.0/2)**2,radians(30.0/2)**2])
        self.control_limit = np.array([[-self.max_throttle,self.max_throttle],[-radians(27.1),radians(27.1)]])

        self.prepareDiscretizedRaceline()
        # describe track boundary as offset from raceline
        self.createBoundary(show=True)

        self.mppi = MPPI(self.samples_count,self.horizon_steps,self.state_dim,self.control_dim,self.temperature,self.mppi_dt,self.noise_cov,self.discretized_raceline,cuda=True,cuda_filename="mppi/mppi_racecar.cu")

        self.mppi.applyDiscreteDynamics = self.applyDiscreteDynamics
        self.mppi.evaluateStepCost = self.evaluateStepCost
        self.mppi.evaluateTerminalCost = self.evaluateTerminalCost

        if (sim is None):
            self.lf = 0.09-0.036
            self.lr = 0.036
            self.L = 0.09
            self.Df = 3.93731
            self.Dr = 6.23597
            self.C = 2.80646
            self.B = 0.51943
            self.Cm1 = 6.03154
            self.Cm2 = 0.96769
            self.Cr = -0.20375
            self.Cd = 0.00000
            self.Iz = 0.00278
            self.m = 0.1667
        else:
            self.lf = sim.lf 
            self.lr = sim.lr 
            self.L = sim.L 
            self.Df = sim.Df 
            self.Dr = sim.Dr 
            self.C = sim.C 
            self.B = sim.B 
            self.Cm1 = sim.Cm1 
            self.Cm2 = sim.Cm2 
            self.Cr = sim.Cr 
            self.Cd = sim.Cd 
            self.Iz = sim.Iz 
            self.m = sim.m 
        return

    def prepareDiscretizedRaceline(self):
        ss = np.linspace(0,self.track.raceline_len_m,self.discretized_raceline_len)
        rr = splev(ss%self.track.raceline_len_m,self.track.raceline_s,der=0)
        drr = splev(ss%self.track.raceline_len_m,self.track.raceline_s,der=1)
        heading_vec = np.arctan2(drr[1],drr[0])
        vv = self.track.sToV(ss)

        # parameter, distance along track
        self.ss = ss
        self.raceline_points = np.array(rr)
        self.raceline_headings = heading_vec
        self.raceline_velocity = vv
        self.discretized_raceline = np.vstack([self.raceline_points,self.raceline_headings,vv]).T
        return

    def createBoundary(self,show=False):
        # construct a (self.discretized_raceline_len * 2) vector
        # to record the left and right track boundary as an offset to the discretized raceline
        left_boundary = []
        right_boundary = []
        for i in range(self.discretized_raceline_len):
            # find normal direction
            coord = self.raceline_points[:,i]
            heading = self.raceline_headings[i]



            left, right = self.track.preciseTrackBoundary(coord,heading)
            print(left,right)
            left_boundary.append(left)
            right_boundary.append(right)

            # DEBUG
            # calculate left/right boundary
            left_point = (coord[0] + left * cos(heading+np.pi/2),coord[1] + left * sin(heading+np.pi/2))
            right_point = (coord[0] + right * cos(heading-np.pi/2),coord[1] + right * sin(heading-np.pi/2))
            img = self.track.drawTrack()
            img = self.track.drawRaceline(img = img)
            img = self.track.drawPoint(img,coord,color=(0,0,0))
            img = self.track.drawPoint(img,left_point,color=(0,0,0))
            img = self.track.drawPoint(img,right_point,color=(0,0,0))
            plt.imshow(img)
            plt.show()




            breakpoint()
            left, right = self.track.preciseTrackBoundary(coord,heading)

        if (show):
            img = self.track.drawTrack()
            img = self.track.drawRaceline(img = img)
            points = np.vstack([left_boundary,right_boundary])
            breakpoint()
            img = self.track.drawPolyline(points,lineColor=(0,255,0),img=img)
            plt.imshow(img)
            plt.show()
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
    def ctrlCar(self,state,track,v_override=None,reverse=False):
        p = self.p
        p.s()
        # get an estimate for current distance along raceline
        debug_dict = {'x_ref_r':[],'x_ref_l':[],'x_ref':[],'crosstrack_error':[],'heading_error':[]}


        try:
            self.predictOpponent()
            debug_dict['opponent'] = self.opponent_prediction
        except AttributeError:
            pass

        #e_cross, e_heading, v_ref, k_ref, coord_ref, valid = track.getRefPoint(state, 3, 0.01, reverse=reverse)
        #debug_dict['crosstrack_error'] = e_cross
        #debug_dict['heading_error'] = e_heading
        p.s("local traj")
        if self.last_s is None:
            retval = track.localTrajectory(state,wheelbase=0.102/2.0,return_u=True)
            if retval is None:
                print_warning("localTrajectory returned None")
                ret =  (0,0,False,debug_dict)
                return ret
            else:
                # parse return value from localTrajectory
                (local_ctrl_pnt,offset,orientation,curvature,v_target,u0) = retval
                self.last_s = track.uToS(u0).item()
        p.e("local traj")

        p.s("prep")
        s0 = self.last_s
        # vehicle state
        # vf: forward positive
        # vs: left positive
        x,y,heading,vf,vs,omega = state
        dx = vf*cos(heading) - vs*sin(heading)
        dy = vf*sin(heading) + vs*cos(heading)

        self.states = np.array([x,dx,y,dy,heading,omega])
        state = np.array([x,dx,y,dy,heading,omega])

        ref_control = np.zeros([self.horizon_steps,self.control_dim])
        p.e("prep")

        p.s("mppi")
        uu = self.mppi.control(state.copy(),self.opponent_prediction,self.control_limit)
        control = uu[0]
        throttle = control[0]
        steering = control[1]
        p.e("mppi")

        # DEBUG
        # simulate where mppi think where the car will end up with
        # with synthesized control sequence
        p.s("debug")
        sim_state = state.copy()
        for i in range(self.horizon_steps):
            sim_state = self.applyDiscreteDynamics(sim_state,uu[i],self.mppi_dt)
            coord = (sim_state[0],sim_state[2])
            debug_dict['x_ref'].append(coord)

        # DEBUG
        # per ji's request, show 100 sampled trajectory, randomly selected
        

        ret =  (throttle,steering,True,debug_dict)
        p.e("debug")
        p.e()
        return ret


    def evaluateStepCost(self,state,control):
        heading = state[4]
        # calculate cost
        # cost = -reward + penalty
        #ids0 = self.findClosestIds(x0)
        ids = self.findClosestIds(state)

        # reward is progress along centerline
        #cost = - ( self.ss[ids] - self.ss[ids0] )

        # determine lateral offset
        cost = np.sqrt((state[0]-self.raceline_points[0,ids])**2+(state[2]-self.raceline_points[1,ids])**2) * 0.5

        # heading error cost
        # cost += abs((self.raceline_headings[ids] - heading + np.pi) % (2*np.pi) - np.pi)
        return cost*10
        # FIXME
        #return 0.0

    def findClosestIds(self,state):
        x = state[0]
        y = state[2]
        dx = x - self.raceline_points[0]
        dy = y - self.raceline_points[1]

        dist2 = dx*dx + dy*dy
        idx = np.argmin(dist2)
        return idx

    def evaluateTerminalCost(self,state,x0):
        heading = state[4]
        # calculate cost
        # cost = -reward + penalty
        ids0 = self.findClosestIds(x0)
        ids = self.findClosestIds(state)

        # reward is progress along centerline
        cost_real = - ( self.ss[ids] - self.ss[ids0] )
        #print("real = %.4f"%(cost_real))
        # instead of actual progress length, can we approximate progress with index to self.ss ?
        # 0.01 is to roughly convert indices difference into path length difference in meter
        cost_approx = -((ids - ids0 + self.discretized_raceline_len)%self.discretized_raceline_len)*0.01
        #print("approx = %.4f"%(cost_approx))
        #print((cost_approx - cost_real)/cost_real)

        # sanity check, 0.5*0.1m offset equivalent to 0.1 rad(5deg) heading error
        # 10cm progress equivalent to 0.1 rad error
        # sounds bout right

        return cost_real*10
        # NOTE ignoring terminal cost
        #return 0.0

    # advance car dynamics
    # for use in visualization
    def applyDiscreteDynamics(self,state,control,dt):
        x = state[0]
        vxg = state[1]
        # left pos(+)
        y = state[2]
        vyg = state[3]
        heading = state[4]
        omega = state[5]

        throttle = control[0]
        # left pos(+)
        steering = control[1]

        lf = self.lf
        lr = self.lr
        L = self.L

        Df = self.Df
        Dr = self.Dr
        B = self.B
        C = self.C
        Cm1 = self.Cm1
        Cm2 = self.Cm2
        Cr = self.Cr
        Cd = self.Cd
        Iz = self.Iz
        m = self.m


        # forward
        vx = vxg*cos(heading) + vyg*sin(heading)
        # lateral, left +
        vy = -vxg*sin(heading) + vyg*cos(heading)

        # for small velocity, use kinematic model 
        if (vx<0.05):
            beta = atan(lr/L*tan(steering))
            norm = lambda a,b:(a**2+b**2)**0.5
            # motor model
            d_vx = (( Cm1 - Cm2 * vx) * throttle - Cr - Cd * vx * vx)
            vx = vx + d_vx * dt
            vy = norm(vx,vy)*sin(beta)
            d_omega = 0.0
            omega = vx/L*tan(steering)

            slip_f = 0
            slip_r = 0
            Ffy = 0
            Fry = 0

        else:
            slip_f = -np.arctan((omega*lf + vy)/vx) + steering
            slip_r = np.arctan((omega*lr - vy)/vx)

            Ffy = Df * np.sin( C * np.arctan(B *slip_f)) * 9.8 * lr / (lr + lf) * m
            Fry = Dr * np.sin( C * np.arctan(B *slip_r)) * 9.8 * lf / (lr + lf) * m

            # motor model
            Frx = (( Cm1 - Cm2 * vx) * throttle - Cr - Cd * vx * vx)*m

            # Dynamics
            d_vx = 1.0/m * (Frx - Ffy * np.sin( steering ) + m * vy * omega)
            d_vy = 1.0/m * (Fry + Ffy * np.cos( steering ) - m * vx * omega)
            d_omega = 1.0/Iz * (Ffy * lf * np.cos( steering ) - Fry * lr)

            # discretization
            vx = vx + d_vx * dt
            vy = vy + d_vy * dt
            omega = omega + d_omega * dt 

        # back to global frame
        vxg = vx*cos(heading)-vy*sin(heading)
        vyg = vx*sin(heading)+vy*cos(heading)

        # apply updates
        # TODO add 1/2 a t2
        x += vxg*dt
        y += vyg*dt
        heading += omega*dt + 0.5* d_omega * dt * dt

        retval = (x,vxg,y,vyg,heading,omega )
        return np.array(retval)


    # we assume opponent will follow reference trajectory at current speed
    def initTrackOpponents(self):
        return

    def predictOpponent(self):
        self.opponent_prediction = []
        for opponent in self.opponents:
            traj = self.track.predictOpponent(opponent.state, self.horizon_steps, self.mppi_dt)
            self.opponent_prediction.append(traj)


if __name__=="__main__":
    pass

        
