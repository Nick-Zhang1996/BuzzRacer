
from common import *
from controller.iLQGameCarController import iLQGameCarController
from controller.LQGame import my_solve_lq_game

class iLQGameSoloCarController(iLQGameCarController):
    def __init__(self, car,config):
        super().__init__(car,config)

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
        for iteration in range(3):
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
            B1s = [np.vstack([Bi,np.zeros((n,m))]) for Bi in Bis]
            B2s = [np.vstack([np.zeros((n,m)),Bj]) for Bj in Bjs]


            Q1s,q1s,Q2s,q2s,Rs,rs = self.getCostMatrices(xx_i,uu_i,xx_j,uu_j)
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
                [Q1s, Q2s], [q1s, q2s], [Rs[0][0], Rs[1][1]],rs,self.lqt)

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

            self.t.e('my_solve_lq_game')

        dx_i = xx_i[0] - self.x_i_ref[0]
        dx_j = xx_j[0] - self.x_j_ref[0]
        dx = np.vstack([dx_i,dx_j])
        ctrl1 = (self.u_i_ref[0] - P1s[0] @ dx + alpha1s[0]).flatten()
        ctrl2 = (self.u_j_ref[0] - P2s[0] @ dx + alpha2s[0]).flatten()

        self.debug_dict.update({'u_ref':np.array(self.u_i_ref), 'x_ref':np.array(self.x_i_ref), 'x_j_ref':np.array(self.x_j_ref), 'u_j_ref':np.array(self.u_j_ref)})
        self.t.e()
        return ctrl1,ctrl2
