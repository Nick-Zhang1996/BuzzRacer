# unused planner cold
    def demoSingleIntegrator(self):
        # state x:[s,n]
        # u: [us=ds,un=dn] first derivative of s and n
        self.n = n = 2
        self.m = m = 2
        # p: prediction horizon
        self.p = p = horizon = 50

        self.genPath()

        x0 = [1,0.15]
        xref = [7,-0.15]

        (u_vec, state_traj) = self.solveSingleIntegrator(x0,xref)
        print("err = %.2f "%(np.linalg.norm(np.array(xref)-state_traj[-1,:])))

        self.plotTrack()
        # initial state
        self.plotCar(x0)
        # target state
        self.plotCar(xref)
        self.plotStateTraj(state_traj)
        plt.axis('equal')
        plt.show()


    def solveSingleIntegrator(self,x0,xref):
        mpc = MPC()
        n = self.n
        m = self.m
        p = self.p
        # setup mpc 
        mpc.setup(n,m,n,p)
        A = np.eye(n)
        dt = 0.1
        B = np.eye(2)*dt
        P = np.diag([1,1])
        Q = np.eye(m)*5.0

        t = time()
        x0 = np.array(x0).T.reshape(-1,1)
        xref = np.array(xref).T.reshape(-1,1)
        xref_vec = xref.repeat(p,1).T.reshape(p,n,1)
        #du_max = np.array([[1,1]]).T
        du_max = None
        #u_max = np.array([[1.5,1.5]]).T
        u_max = None
        mpc.convertLtiPlanner(A,B,P,Q,xref_vec,x0,p,u_max,du_max)
        # add track boundary constraints
        self.constructStateLimits(mpc)
        mpc.solve()
        dt = time()-t
        if (mpc.h is not None):
            print(mpc.h.shape)
        print("freq = %.2fHz"%(1/dt))
        # state_traj in curvilinear frame
        state_traj = mpc.F @ mpc.u + mpc.Ex0
        state_traj = state_traj.reshape((p,n))
        state_traj = np.vstack([x0.T,state_traj])

        '''
        plt.plot(state_traj)
        plt.show()
        u = np.array(mpc.u).reshape((-1,2))
        plt.plot(u)
        plt.show()
        '''
        return (mpc.u,state_traj)

    def demoDoubleIntegrator(self):
        # state x:[s,n,ds,dn]
        # u: [us=dds,un=ddn] second derivative of s and n
        self.n = n = 4
        self.m = m = 2
        # p: prediction horizon
        self.p = p = horizon = 50

        self.genPath()

        x0 = [1,0.15,0,0]
        xref = [7,-0.15,0,0]

        (u_vec, state_traj) = self.solveDoubleIntegrator(x0,xref)
        '''
        print("u_vec")
        print(u_vec)
        print("state_traj")
        print(state_traj)
        '''
        print("err = %.2f "%(np.linalg.norm(np.array(xref)-state_traj[-1,:])))

        self.plotTrack()
        # initial state
        self.plotCar(x0)
        # target state
        self.plotCar(xref)
        self.plotStateTraj(state_traj)
        plt.axis('equal')
        plt.show()


    def solveDoubleIntegrator(self,x0,xref):
        mpc = MPC()
        n = self.n
        m = self.m
        p = self.p
        # setup mpc 
        mpc.setup(n,m,n,p)
        A = np.eye(4)
        dt = 0.1
        A[0,2] = dt
        A[1,3] = dt
        B = np.array([[0.5*dt**2,0],[0,0.5*dt**2],[dt,0],[0,dt]])
        P = np.diag([1,1,1,1])
        Q = np.eye(m)*0.5

        t = time()
        x0 = np.array(x0).T.reshape(-1,1)
        xref = np.array(xref).T.reshape(-1,1)
        xref_vec = xref.repeat(p,1).T.reshape(p,n,1)
        #du_max = np.array([[1,1]]).T
        du_max = None
        #u_max = np.array([[1.5,1.5]]).T
        u_max = None
        mpc.convertLtiPlanner(A,B,P,Q,xref_vec,x0,p,u_max,du_max)
        # add track boundary constraints
        self.constructStateLimits(mpc)
        mpc.solve()
        dt = time()-t
        print(mpc.h.shape)
        print("freq = %.2fHz"%(1/dt))
        # state_traj in curvilinear frame
        state_traj = mpc.F @ mpc.u + mpc.Ex0
        state_traj = state_traj.reshape((p,n))
        state_traj = np.vstack([x0.T,state_traj])
        '''
        plt.plot(state_traj)
        plt.show()
        u = np.array(mpc.u).reshape((-1,2))
        plt.plot(u)
        plt.show()
        '''
        return (mpc.u,state_traj)

