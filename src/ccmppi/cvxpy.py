# not a real python script
# just storing cvxpy implementation of cc

    # apply covariance control
    #
    # input:
    #   state: (x,y,v,heading)
    # return: N K matrices of size (n,m)
    # this new version uses given reference trajectory, from optimum trajectory in last solution
    # ref_state_vec: N*[x,y,v,heading]
    # ref_ctrl_vec: N*[throttle, steering]
    # cvxpy version, new formulation
    def cc_cvxpy(self, state, ref_state_vec, ref_ctrl_vec, return_sx=False, debug=False):
        n = self.n
        N = self.N
        m = self.m
        l = self.l

        # find where the car is in reference to reference trajectory
        ref_xx = ref_state_vec[:,0]
        ref_yy = ref_state_vec[:,1]

        x,y,_,_ = state

        dist_sqr = (ref_xx-x)**2 + (ref_yy-y)**2
        # start : index of closest ref point to car
        start = np.argmin(dist_sqr)

        # As = [A0..A(N-1)]
        # linearize dynamics around ref traj
        # As = [A0..A(N-1)]
        As = []
        Bs = []
        ds = []
        for i in range(self.N):
            # NOTE this gives discretized dynamics
            A,B,d = self.linearize(ref_state_vec[i,:],ref_ctrl_vec[i,:])
            As.append(A)
            Bs.append(B)
            ds.append(d)


        # assemble big matrices for batch dynamics
        self.As = As = np.dstack(As)
        self.Bs = Bs = np.dstack(Bs)
        self.ds = ds = np.dstack(ds).reshape((self.n,1,self.N))

        # NOTE ds, the offset,  is calculated off reference trajectory
        # additional offsert may need to be added to account for difference between
        # actual state and reference state
        #state_diff = state - self.ref_state_vec[0]
        #state_diff = state_diff.reshape(4,1)
        #ds[:3,:,0] += state_diff[:3]
        #ds[:2,:,0] += state_diff[:2]
        #ds[:,:,0] += state_diff

        if (debug):
            print_info("[cc] ref state x0 (x,y,v,heading)")
            print(ref_state_vec[0])
            print_info("[cc] actual state x0")
            print(state)
            #print_info("[cc] state diff")
            #print(state_diff.flatten())

        A, B, C, d, D = self.make_batch_dynamics(As, Bs, ds, None, self.Sigma_epsilon)

        # cost matrix 
        #Q = np.eye(n)
        #Q_bar = np.kron(np.eye(N+1, dtype=int), Q)
        # soft constraint Q matrix
        Q_bar = np.zeros([(N+1)*self.n, (N+1)*self.n])
        #Q_bar[-self.n:, -self.n:] = np.eye(self.n) * 3000
        Q_bar[-self.n:, -self.n:] = np.eye(self.n) * self.Qf

        R = np.eye(m)
        R_bar = np.kron(np.eye(N, dtype=int), R)

        # technically incorrect, but we can just specify R_bar_1/2 instead of R_bar
        R_bar_sqrt = R_bar
        Q_bar_sqrt = Q_bar

        # terminal covariance constrain
        # not needed with soft constraint
        #sigma_f = self.sigma_f

        # setup cvxpy
        I = np.eye(n*(N+1))
        E_N = np.zeros((n,n*(N+1)))
        E_N[:,n*(N):] = np.eye(n)

        # assemble K as a diagonal block matrix with K_0..K_N-1 as var
        Ks = [cp.Variable((m,n)) for i in range(N)]
        # K dim: mN x n(N+1)
        K = cp.hstack([Ks[0], np.zeros((m,(N)*n))])
        for i in range(1,N):
            line = cp.hstack([ np.zeros((m,n*i)), Ks[i], np.zeros((m,(N-i)*n)) ])
            K = cp.vstack([K, line])

        #objective = cp.Minimize(cp.norm(cp.vec(R_bar_sqrt @ K @ D)) + cp.norm(cp.vec(Q_bar_sqrt @ (I + B@K) @ D )))
        # new formulation
        vecK = cp.vec(K)
        obj = gurobi_trAXB(D.T @ Q_bar_sqrt @ Q_bar_sqrt @ B, D, vecK)
        obj += gurobi_matrix_quad(D, Q_bar_sqrt @ B, vecK)
        #obj += D.T @ Q_bar_sqrt @ Q_bar_sqrt @ D
        obj += gurobi_trAXB( D.T @ Q_bar_sqrt @ Q_bar_sqrt @ B, D, vecK)
        obj += gurobi_matrix_quad(D, R_bar_sqrt @ R_bar_sqrt, vecK)

        #sigma_y_sqrt = self.nearest_spd_cholesky(D@D.T)
        # hard constraint, cvxpy doesn't respect this for some reasons
        #constraints = [cp.bmat([[sigma_f, E_N @(I+B@K)@sigma_y_sqrt], [ sigma_y_sqrt@(I+B @ K).T@E_N.T, I ]]) >= 0]
        constraints = []
        prob = cp.Problem(cp.Minimize(obj), constraints)

        J = prob.solve()

        Ks = np.array([val.value for val in Ks])

        if (debug):
            print_info("[cc] Problem status")
            print(prob.status)
            
        # DEBUG veirfy constraint
        '''
        test_mtx = np.block([[sigma_f, E_N @(I+B@K.value)@sigma_y_sqrt], [ sigma_y_sqrt@(I+B @ K.value).T@E_N.T, I ]])
        if not (np.all(np.linalg.eigvals(test_mtx) > 0)):
            print_warning("[cc] constraint not satisfied")
        '''

        self.Ks = Ks

        As = np.swapaxes(As,0,2)
        As = np.swapaxes(As,1,2)

        Bs = np.swapaxes(Bs,0,2)
        Bs = np.swapaxes(Bs,1,2)

        ds = np.swapaxes(ds,0,2)
        ds = np.swapaxes(ds,1,2)

        # return terminal covariance, theoretical values with and without cc
        if (return_sx):
            reconstruct_K = np.hstack([Ks[0], np.zeros((m,(N)*n))])
            for i in range(1,N):
                line = np.hstack([ np.zeros((m,n*i)), Ks[i], np.zeros((m,(N-i)*n)) ])
                reconstruct_K = np.vstack([reconstruct_K, line])
            Sigma_0 = np.zeros([n,n])
            #Sx_cc = (I + B@K.value ) @ (A @ Sigma_0 @ A.T + D @ D.T ) @ (I + B@K.value ).T
            Sx_cc = (I + B@reconstruct_K ) @ (A @ Sigma_0 @ A.T + D @ D.T ) @ (I + B@reconstruct_K ).T
            Sx_nocc = (A @ Sigma_0 @ A.T + D @ D.T )
            return Ks, As, Bs, ds, Sx_cc, Sx_nocc
        else:
            return Ks, As, Bs, ds

    # apply covariance control
    #
    # input:
    #   state: (x,y,v,heading)
    # return: N K matrices of size (n,m)
    # this new version uses given reference trajectory, from optimum trajectory in last solution
    # ref_state_vec: N*[x,y,v,heading]
    # ref_ctrl_vec: N*[throttle, steering]
    # cvxpy version
    def cvxpy_cc(self, state, ref_state_vec, ref_ctrl_vec, return_sx=False, debug=False):
        n = self.n
        N = self.N
        m = self.m
        l = self.l

        # find where the car is in reference to reference trajectory
        ref_xx = ref_state_vec[:,0]
        ref_yy = ref_state_vec[:,1]

        x,y,_,_ = state

        dist_sqr = (ref_xx-x)**2 + (ref_yy-y)**2
        # start : index of closest ref point to car
        start = np.argmin(dist_sqr)

        # As = [A0..A(N-1)]
        # linearize dynamics around ref traj
        # As = [A0..A(N-1)]
        As = []
        Bs = []
        ds = []
        for i in range(self.N):
            # NOTE this gives discretized dynamics
            A,B,d = self.linearize(ref_state_vec[i,:],ref_ctrl_vec[i,:])
            As.append(A)
            Bs.append(B)
            ds.append(d)


        # assemble big matrices for batch dynamics
        self.As = As = np.dstack(As)
        self.Bs = Bs = np.dstack(Bs)
        self.ds = ds = np.dstack(ds).reshape((self.n,1,self.N))

        # NOTE ds, the offset,  is calculated off reference trajectory
        # additional offsert may need to be added to account for difference between
        # actual state and reference state
        #state_diff = state - self.ref_state_vec[0]
        #state_diff = state_diff.reshape(4,1)
        #ds[:3,:,0] += state_diff[:3]
        #ds[:2,:,0] += state_diff[:2]
        #ds[:,:,0] += state_diff

        if (debug):
            print_info("[cc] ref state x0 (x,y,v,heading)")
            print(ref_state_vec[0])
            print_info("[cc] actual state x0")
            print(state)
            #print_info("[cc] state diff")
            #print(state_diff.flatten())

        A, B, C, d, D = self.make_batch_dynamics(As, Bs, ds, None, self.Sigma_epsilon)

        # cost matrix 
        #Q = np.eye(n)
        #Q_bar = np.kron(np.eye(N+1, dtype=int), Q)
        # soft constraint Q matrix
        Q_bar = np.zeros([(N+1)*self.n, (N+1)*self.n])
        #Q_bar[-self.n:, -self.n:] = np.eye(self.n) * 3000
        Q_bar[-self.n:, -self.n:] = np.eye(self.n) * self.Qf

        R = np.eye(m)
        R_bar = np.kron(np.eye(N, dtype=int), R)

        # technically incorrect, but we can just specify R_bar_1/2 instead of R_bar
        R_bar_sqrt = R_bar
        Q_bar_sqrt = Q_bar

        # terminal covariance constrain
        # not needed with soft constraint
        #sigma_f = self.sigma_f

        # setup cvxpy
        I = np.eye(n*(N+1))
        E_N = np.zeros((n,n*(N+1)))
        E_N[:,n*(N):] = np.eye(n)

        # assemble K as a diagonal block matrix with K_0..K_N-1 as var
        Ks = [cp.Variable((m,n)) for i in range(N)]
        # K dim: mN x n(N+1)
        K = cp.hstack([Ks[0], np.zeros((m,(N)*n))])
        for i in range(1,N):
            line = cp.hstack([ np.zeros((m,n*i)), Ks[i], np.zeros((m,(N-i)*n)) ])
            K = cp.vstack([K, line])

        objective = cp.Minimize(cp.norm(cp.vec(R_bar_sqrt @ K @ D)) + cp.norm(cp.vec(Q_bar_sqrt @ (I + B@K) @ D )))

        # TODO verify with Ji
        sigma_y_sqrt = self.nearest_spd_cholesky(D@D.T)
        # hard constraint, cvxpy doesn't respect this for some reasons
        #constraints = [cp.bmat([[sigma_f, E_N @(I+B@K)@sigma_y_sqrt], [ sigma_y_sqrt@(I+B @ K).T@E_N.T, I ]]) >= 0]
        constraints = []
        prob = cp.Problem(objective, constraints)

        J = prob.solve()

        Ks = np.array([val.value for val in Ks])

        if (debug):
            print_info("[cc] Problem status")
            print(prob.status)
            
        # DEBUG veirfy constraint
        '''
        test_mtx = np.block([[sigma_f, E_N @(I+B@K.value)@sigma_y_sqrt], [ sigma_y_sqrt@(I+B @ K.value).T@E_N.T, I ]])
        if not (np.all(np.linalg.eigvals(test_mtx) > 0)):
            print_warning("[cc] constraint not satisfied")
        '''

        self.Ks = Ks

        As = np.swapaxes(As,0,2)
        As = np.swapaxes(As,1,2)

        Bs = np.swapaxes(Bs,0,2)
        Bs = np.swapaxes(Bs,1,2)

        ds = np.swapaxes(ds,0,2)
        ds = np.swapaxes(ds,1,2)

        # return terminal covariance, theoretical values with and without cc
        if (return_sx):
            reconstruct_K = np.hstack([Ks[0], np.zeros((m,(N)*n))])
            for i in range(1,N):
                line = np.hstack([ np.zeros((m,n*i)), Ks[i], np.zeros((m,(N-i)*n)) ])
                reconstruct_K = np.vstack([reconstruct_K, line])
            Sigma_0 = np.zeros([n,n])
            #Sx_cc = (I + B@K.value ) @ (A @ Sigma_0 @ A.T + D @ D.T ) @ (I + B@K.value ).T
            Sx_cc = (I + B@reconstruct_K ) @ (A @ Sigma_0 @ A.T + D @ D.T ) @ (I + B@reconstruct_K ).T
            Sx_nocc = (A @ Sigma_0 @ A.T + D @ D.T )
            return Ks, As, Bs, ds, Sx_cc, Sx_nocc
        else:
            return Ks, As, Bs, ds
