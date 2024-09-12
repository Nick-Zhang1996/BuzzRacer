from common import *
from planner.Planner import Planner
from controller.ResidualGameCarController import ResidualGameCarController
from simulator.CurvilinearSimulator import CurvilinearSimulator
from math import sin,cos,atan2
import threading 
import multiprocessing
from time import time,sleep

class ResidualGamePlanner(Planner,ResidualGameCarController):
    def __init__(self,config=None):
        self.config = config
        self.car = None
        self.main = None
        Planner.__init__(self,config)
        self.ego_traj = None
        self.oppo_traj = None
        self.planner_thread = None
        self.has_new_plan = threading.Event()
        self.retval = False


    def init(self):
        ResidualGameCarController.__init__(self,self.car,self.config)
        ResidualGameCarController.preInit(self)
        ResidualGameCarController.init(self)
        self.u_ref = self.residual_game.guess
        self.planner_thread = threading.Thread(name='planner', target=self._threadPlan)
        self.planner_thread.start()

        return

    # read latest states, do planning, and update the reference trajectory
    def _threadPlan(self):
        t_vec = []
        while not self.main.exit_request.is_set():
            if (not self.main.track.isInPassingZone(self.car.states)):
                sleep(0.1)
                continue
            t0 = time()
            parent_pipe, child_pipe = multiprocessing.Pipe()
            process = multiprocessing.Process(name='planner_process', target=self._processPlan, args=(child_pipe,))
            process.start()
            while (not parent_pipe.poll(0.1) and not self.main.exit_request.is_set):
                True
            if (parent_pipe.poll(0.1)):
                xx_i_traj, xx_j_traj, has_converged = parent_pipe.recv()
            process.join()
            #self.retval = self.plan()
            self.retval = has_converged
            if (has_converged):
                self.ego_traj =  xx_i_traj
                self.oppo_traj = xx_j_traj
                self.has_new_plan.set()
            t_vec.append(time()-t0)
            self.print_info(f'mean planner update freq {1/np.mean(t_vec)}')

    def _processPlan(self, out_pipe):
        retval = self.plan()
        out_pipe.send(retval)
        return

    # create a plan, store states internally
    def plan(self):
        x0 = CurvilinearSimulator.cart2CurvTrack(self.ego_car.states, self.main.track)
        x1 = CurvilinearSimulator.cart2CurvTrack(self.oppo_car.states, self.main.track)
        xi_ref, xj_ref, u_ref, has_converged = ResidualGameCarController.solveGame(self,x0,x1, u_ref = self.u_ref)
        self.guess = u_ref

        # convert to cartesian coord
        xx_i_cart = [CurvilinearSimulator.curv2CartTrack(val, self.main.track) for val in xi_ref]
        xx_j_cart = [CurvilinearSimulator.curv2CartTrack(val, self.main.track) for val in xj_ref]

        # catch crazy trajectories
        ego_traj =  np.array(xx_i_cart)
        oppo_traj = np.array(xx_j_cart)
        ego_traj_omega = (np.diff(np.arctan2( np.diff(ego_traj[:,1]),np.diff(ego_traj[:,0]) )) + np.pi ) % (2*np.pi) - np.pi
        oppo_traj_omega = (np.diff(np.arctan2( np.diff(oppo_traj[:,1]),np.diff(oppo_traj[:,0]) )) + np.pi ) % (2*np.pi) - np.pi

        '''
        if (np.any(np.abs(ego_traj_omega) > 0.3) or np.any(np.abs(oppo_traj_omega) > 0.3) ):
            self.print_info('---------------- bad plan -----------')
            return np.array(xx_i_cart), np.array(xx_j_cart), False
        '''


        # store for use in localTrajectory
        # NOTE that when this function is run in a separate process this cannot be counted on
        self.ego_traj =  np.array(xx_i_cart)
        self.oppo_traj = np.array(xx_j_cart)
        return np.array(xx_i_cart), np.array(xx_j_cart), True

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

    def localTrajectory(self,state):
        self_curv_state = CurvilinearSimulator.cart2CurvTrack(self.ego_car.states,self.main.track)
        oppo_curv_state = CurvilinearSimulator.cart2CurvTrack(self.oppo_car.states,self.main.track)
        raceline_point,offset,raceline_orientation,signed_curvature,_ = self.localTrajectoryFromTraj(state,self.ego_traj)
        v_target  = self.main.track.sToV(self_curv_state[0]%self.main.track.raceline_len_m)
        track_len = self.main.track.raceline_len_m

        lead = (self_curv_state[0] - oppo_curv_state[0] + track_len/2)%track_len - track_len/2
        # not in passing zone, too close, stil faster
        if (not self.main.track.isInPassingZone(state) and lead < 0 and lead > -0.3):
            v_target = min(v_target,oppo_curv_state[1]) - 0.3
            self.print_info(f'oppo car keeping back lead = {lead}')
        return raceline_point,offset,raceline_orientation,signed_curvature,v_target

    def oppoLocalTrajectory(self,state):
        self_curv_state = CurvilinearSimulator.cart2CurvTrack(self.ego_car.states,self.main.track)
        oppo_curv_state = CurvilinearSimulator.cart2CurvTrack(self.oppo_car.states,self.main.track)
        raceline_point,offset,raceline_orientation,signed_curvature,_ = self.localTrajectoryFromTraj(state,self.oppo_traj)
        v_target  = self.main.track.sToV(self_curv_state[0]%self.main.track.raceline_len_m)
        track_len = self.main.track.raceline_len_m

        if (not self.main.track.isInPassingZone(state)):
            raceline_point,offset,raceline_orientation,signed_curvature,_ = self.track.localTrajectory(state)


        return raceline_point,offset,raceline_orientation,signed_curvature,v_target




