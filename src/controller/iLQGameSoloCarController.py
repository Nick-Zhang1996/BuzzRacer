
from common import *
from controller.iLQGameCarController import iLQGameCarController
from controller.LQGame import my_solve_lq_game
from scipy.linalg import block_diag

class iLQGameSoloCarController(iLQGameCarController):
    def __init__(self, car,config):
        super().__init__(car,config)
        self.iterations = 1

    # ego car solely responsible for evading opponent car
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
        II = np.hstack([np.eye(n),-np.eye(n)])
        R0 = np.zeros((m,m))

        for t in range(self.horizon):
            # these cost matrices work on the stacked agent state x, not state perturbation dx
            # cost_i = 1/2 x.T @ Qi_x @ x + qi_x.T @ x + 1/2 ui.T @ R @ ui + ri.T @ ui
            Q1_x =  block_diag(self.Q1,np.zeros((n,n)))
            q1_x = np.hstack([self.q1.T,np.zeros((1,n))])
            q1_x[0,n] = self.Qop1

            Q2_x =  block_diag(np.zeros((n,n)),self.Q2)
            q2_x = np.hstack([np.zeros((1,n)),self.q2.T])
            q2_x[0,0] = self.Qop2

            # barrier function: opponent collision
            delta_x = xx_i[t] - xx_j[t]
            if (np.abs(delta_x[0])<1.5*self.opponent_min_distance_s and np.abs(delta_x[2])<1.5*self.opponent_min_distance_n):
                #self.print_info('collision avoidance')
                sgn_s = -1 if delta_x[0]>0 else 1
                sgn_n = -1 if delta_x[2]>0 else 1
                Q1_x += 2* II.T @ self.Qcol @ II
                q1_x += (np.array([[sgn_s*2*self.opponent_min_distance_s, 0, sgn_n*2*self.opponent_min_distance_n, 0]]) @ self.Qcol @ II)

            # barrier function: track boundary
            cart_states_i = self.simulator.curv2Cart(xx_i[t].flatten())
            self.t.s('preciseTrackBoundary')
            left_boundary_i, right_boundary_i = self.main.track.preciseTrackBoundary(cart_states_i[:2],cart_states_i[2])
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
            R1 = 0.005*np.diag([1.0,1.0])
            R2 = 0.005*np.diag([1.0,1.0])
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
