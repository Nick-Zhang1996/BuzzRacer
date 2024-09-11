from common import *
from planner.Planner import Planner
from controller.iLQGameCarController import iLQGameCarController
from simulator.CurvilinearSimulator import CurvilinearSimulator
from math import atan2,sin,cos

class iLQGamePlanner(Planner,iLQGameCarController):
    def __init__(self,config=None):
        self.config = config
        self.car = None
        self.main = None
        Planner.__init__(self,config)
        self.ego_traj = None
        self.oppo_traj = None


    def init(self):
        iLQGameCarController.__init__(self,self.car,self.config)
        iLQGameCarController.preInit(self)

        for car in self.main.cars:
            car.sim_states = CurvilinearSimulator.cart2CurvTrack(car.states,self.main.track)
        iLQGameCarController.init(self)

        if (self.control_opponent):
            self.oppo_car.controller.planner.localTrajectory = self.localTrajectoryOpponent
        return

    # create a plan, store states internally
    def plan(self):
        if (not self.needReplan()):
            return True
        # set all car sim_states
        x0 = CurvilinearSimulator.cart2CurvTrack(self.ego_car.states,self.main.track)
        x1 = CurvilinearSimulator.cart2CurvTrack(self.oppo_car.states,self.main.track)
        # NOTE
        self.ego_car.sim_states = x0
        self.oppo_car.sim_states = x1
        self.print_info(f'v0: {x0[1]:.2f}, v1:{x1[1]:.2f}')

        x0[1] = max(2.0,x0[1])
        x1[1] = max(2.0,x1[1])
        alpha1s, P1s, alpha2s, P2s = iLQGameCarController.lqControl(self,x0,x1)

        # propagate control forward
        xx_i =[x0.reshape((self.n,1))]
        xx_j =[x1.reshape((self.n,1))]
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
            u = np.array(u).reshape(-1,1)
            new_x = self.update_dynamics(xx_i[-1],u)
            xx_i.append(new_x.reshape(4,1))

            # for agent j
            u = self.u_j_ref[t] - P2s[t] @ dx + alpha2s[t]
            u = np.array(u).reshape(-1,1)
            new_x = self.update_dynamics(xx_j[-1],u)
            xx_j.append(new_x.reshape(4,1))
        # convert to cartesian coord
        xx_i_cart = [CurvilinearSimulator.curv2CartTrack(val,self.main.track) for val in xx_i]
        xx_j_cart = [CurvilinearSimulator.curv2CartTrack(val,self.main.track) for val in xx_j]

        # store for use in localTrajectory
        self.ego_traj =  np.array(xx_i_cart)
        self.oppo_traj = np.array(xx_j_cart)

        self.plan_traj = self.ego_traj
        if (self.control_opponent):
            self.oppo_car.controller.planner.plan_traj = self.oppo_traj
        return True

    def plotDebug(self):
        #plot debug information
        if (self.main.visualization.update_visualization.is_set()):
            img = self.main.visualization.visualization_img
            ego_path = [val[:2] for val in self.ego_traj]
            oppo_path = [val[:2] for val in self.oppo_traj]
            img = self.main.track.drawPolyline(ego_path,img)
            img = self.main.track.drawPolyline(oppo_path,img)
            self.main.visualization.visualization_img = img
        return

    def localTrajectoryFromTraj(self,state,traj):
        #(local_ctrl_pnt,offset,orientation,curvature,v_target) = retval
        #(_,offset,orientation,_,v_target) = retval
        x = state[0]
        y = state[1]
        heading = state[2]
        vf = state[3]
        vs = state[4]
        omega = state[5]

        # find the coordinate of center of front axle
        wheelbase = 0.1
        x += wheelbase*cos(heading)
        y += wheelbase*sin(heading)

        dxx = traj[:-1,0]-x
        dyy = traj[:-1,1]-y
        index = np.argmin(dxx**2+dyy**2)
        raceline_point = (traj[index,:2])

        # find offset
        # positive offset means car is to the left of the trajectory(need to turn right)
        dr = traj[index+1,:2] - traj[index,:2]
        track_to_car = (x-traj[index,0], y-traj[index,1])
        offset = np.cross(dr/np.linalg.norm(dr),track_to_car).item()

        raceline_orientation = atan2(dr[1],dr[0])

        #signed_curvature = splev(self.ss[index],self.curvature_fun)[0].item()
        signed_curvature = 0


        # reference point on raceline,lateral offset, tangent line orientation, curvature(signed, ccw+), recommended velocity
        return (raceline_point,offset,raceline_orientation,signed_curvature,None)

    def localTrajectoryOpponent(self,state):
        self_curv_state = self.oppo_car.sim_states
        oppo_curv_state = self.ego_car.sim_states
        raceline_point,offset,raceline_orientation,signed_curvature,_ = self.localTrajectoryFromTraj(state,self.oppo_traj)
        v_target  = self.main.track.sToV(oppo_curv_state[0]%self.main.track.raceline_len_m)
        if (v_target > 1.0):
            v_target = 1.0 + (v_target-1.0)*0.5
        track_len = self.main.track.raceline_len_m
        lead = (self_curv_state[0] - oppo_curv_state[0] + track_len/2)%track_len - track_len/2
        v_diff = self_curv_state[1] - oppo_curv_state[1]
        # not in passing zone, too close, stil faster
        if (not self.main.track.isInPassingZone(state) and lead < 0 and lead > -0.4):
            self.print_info(f'oppo car keeping back lead = {lead}')
            v_target = min(v_target,oppo_curv_state[1]) - 0.3
        return raceline_point,offset,raceline_orientation,signed_curvature,v_target

    def localTrajectory(self,state):
        self_curv_state = self.ego_car.sim_states
        oppo_curv_state = self.oppo_car.sim_states

        raceline_point,offset,raceline_orientation,signed_curvature,_ = self.localTrajectoryFromTraj(state,self.ego_traj)
        v_target  = self.main.track.sToV(self_curv_state[0]%self.main.track.raceline_len_m)
        track_len = self.main.track.raceline_len_m
        lead = (self_curv_state[0] - oppo_curv_state[0] + track_len/2)%track_len - track_len/2
        v_diff = self_curv_state[1] - oppo_curv_state[1]
        # not in passing zone, too close, stil faster
        if (not self.main.track.isInPassingZone(state) and lead < 0 and lead > -0.3):
            self.print_info(f'ego car keeping back lead = {lead}')
            v_target = min(v_target,oppo_curv_state[1]) - 0.3
        return raceline_point,offset,raceline_orientation,signed_curvature,v_target


