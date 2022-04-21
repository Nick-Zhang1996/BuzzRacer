# linear time variant mpc with gurobi
import gurobipy as gp
from gurobipy import GRB
from timeUtil import *

import numpy as np
import scipy.sparse as sp

class MPC:
    # x_bound: n*2, min,max
    def __init__(self, state_dim, control_dim, horizon, dt, x_bound, u_bound):
        self.horizon = horizon
        self.dt = dt
        # control dimension
        self.m = control_dim
        # state dimension
        self.n = state_dim
        self.x_bound = x_bound
        self.u_bound = u_bound
        self.gurobi_constrain_x0 = None
        self.t = execution_timer(True)

    def buildModel(self, As, Bs, Qs, Rs):
        N = self.horizon

        As = np.array(As)
        assert (As.shape == (N,self.n,self.n))
        Bs = np.array(Bs)
        assert (Bs.shape == (N,self.n,self.m))
        Qs = np.array(Qs)
        assert (Qs.shape == (N,self.n,self.n))

        Rs = np.array(Rs)
        assert (Rs.shape == (N,self.m,self.m))

        m = gp.Model("mpc_inc")
        m.setParam(GRB.Param.OutputFlag, 0)
        xmin = np.tile(self.x_bound[:,0].reshape((-1,1)), (1,N))
        xmax = np.tile(self.x_bound[:,1].reshape((-1,1)), (1,N))
        umin = np.tile(self.u_bound[:,0].reshape((-1,1)), (1,N))
        umax = np.tile(self.u_bound[:,1].reshape((-1,1)), (1,N))

        # x1..xN
        x = m.addMVar(shape=(self.n, N), lb=xmin, ub=xmax, name='x')
        u = m.addMVar(shape=(self.m, N), lb=umin, ub=umax, name='u')
        self.gurobi_x = x
        self.gurobi_u = u

        # calculate G from A, use taylor expansion
        # G = I + A*dt
        # calculate H from B, use taylor expansion
        # H = B*dt
        Gs = []
        Hs = []
        for i in range(N):
            Gs.append( np.eye(self.n) + As[i] * self.dt )
            Hs.append( Bs[i] * self.dt)

        for i in range(1,N):
            # xi = Gi*x(i-1) + Hi*ui
            m.addConstr( x[:,i] == Gs[i] @ x[:,i-1] + Hs[i] @ u[:,i] )
        self.model = m
        self.Gs = Gs
        self.Hs = Hs
        self.Qs = Qs
        self.Rs = Rs
        return

    def step(self, x0, x_ref):
        t = self.t
        t.s()
        N = self.horizon
        m = self.model
        x0 = np.array(x0).reshape((-1,1))
        assert (x0.shape == (self.n,1))
        x_ref = np.array(x_ref)
        assert (x_ref.shape == (self.n, N))

        # update x0 dependent constrain
        # x1 = G1*x0 + H1*u1
        x = self.gurobi_x
        u = self.gurobi_u
        Gs = self.Gs
        Hs = self.Hs
        Qs = self.Qs
        Rs = self.Rs

        t.s("constrain")
        if (not (self.gurobi_constrain_x0 is None)):
            self.model.remove(self.gurobi_constrain_x0)
        self.gurobi_constrain_x0= m.addConstr( x[:,0] == (Gs[0] @ x0).flatten() + Hs[0] @ u[:,0] )
        t.e("constrain")

        #objective_xQx = sum((x[:,i]-x_ref[:,i]) @ Qs[i] @ (x[:,i]-x_ref[:,i]) for i in range(N))
        # convert above to following valid form for Gurobi
        #objective_xQx = sum(x[:,i] @ Qs[i] @ x[:,i] - 2*x_ref[:,i] @ Qs[i] @ x[:,i] + x_ref[:,i] @ Qs[i] @ x_ref[:,i] for i in range(N))
        t.s('assm objective')
        i = N-1
        objective_xQx = x[:,i] @ Qs[i] @ x[:,i] - 2*x_ref[:,i] @ Qs[i] @ x[:,i] + x_ref[:,i] @ Qs[i] @ x_ref[:,i]
        t.e('assm objective')
        #objective_uRu = sum(u[:,i] @ Rs[i] @ u[:,i] for i in range(N))
        #m.setObjective(objective_xQx + objective_uRu, GRB.MINIMIZE)
        t.s('set obj')
        m.setObjective(objective_xQx , GRB.MINIMIZE)
        t.e('set obj')
        t.s('optimize')
        m.optimize()
        t.e('optimize')
        if (m.status != GRB.OPTIMAL):
            print("optimal solution NOT found, status code=%d"%m.status)
        print("objective val = %.4f"%m.objVal)
        t.e()
        return (x.x,u.x)


    def control_ltv(self, x0,x_ref, As, Bs, Qs, Rs):
        t = self.t

        t.s()
        N = self.horizon

        x0 = np.array(x0).reshape((-1,1))
        assert (x0.shape == (self.n,1))
        x_ref = np.array(x_ref)
        assert (x_ref.shape == (self.n, N))
        As = np.array(As)
        assert (As.shape == (N,self.n,self.n))
        Bs = np.array(Bs)
        assert (Bs.shape == (N,self.n,self.m))
        Qs = np.array(Qs)
        assert (Qs.shape == (N,self.n,self.n))
        Rs = np.array(Rs)
        assert (Rs.shape == (N,self.m,self.m))

        m = gp.Model("mpc")
        m.setParam(GRB.Param.OutputFlag, 0)
        xmin = np.tile(self.x_bound[:,0].reshape((-1,1)), (1,N))
        xmax = np.tile(self.x_bound[:,1].reshape((-1,1)), (1,N))
        umin = np.tile(self.u_bound[:,0].reshape((-1,1)), (1,N))
        umax = np.tile(self.u_bound[:,1].reshape((-1,1)), (1,N))

        # x1..xN
        x = m.addMVar(shape=(self.n, N), lb=xmin, ub=xmax, name='x')
        u = m.addMVar(shape=(self.m, N), lb=umin, ub=umax, name='u')
        

        # calculate G from A, use taylor expansion
        # G = I + A*dt
        # calculate H from B, use taylor expansion
        # H = B*dt
        Gs = []
        Hs = []
        for i in range(N):
            Gs.append( np.eye(self.n) + As[i] * self.dt )
            Hs.append( Bs[i] * self.dt)

        t.s("constrain")
        # x1 = G1*x0 + H1*u1
        m.addConstr( x[:,0] == (Gs[0] @ x0).flatten() + Hs[0] @ u[:,0] )
        for i in range(1,N):
            # xi = Gi*x(i-1) + Hi*ui
            m.addConstr( x[:,i] == Gs[i] @ x[:,i-1] + Hs[i] @ u[:,i] )
        t.e("constrain")

        #objective_xQx = sum((x[:,i]-x_ref[:,i]) @ Qs[i] @ (x[:,i]-x_ref[:,i]) for i in range(N))
        # convert above to following valid form for Gurobi
        #objective_xQx = sum(x[:,i] @ Qs[i] @ x[:,i] - 2*x_ref[:,i] @ Qs[i] @ x[:,i] + x_ref[:,i] @ Qs[i] @ x_ref[:,i] for i in range(N))
        i = N-1
        t.s("obj")
        objective_xQx = x[:,i] @ Qs[i] @ x[:,i] - 2*x_ref[:,i] @ Qs[i] @ x[:,i] + x_ref[:,i] @ Qs[i] @ x_ref[:,i]
        #objective_uRu = sum(u[:,i] @ Rs[i] @ u[:,i] for i in range(N))
        #m.setObjective(objective_xQx + objective_uRu, GRB.MINIMIZE)
        m.setObjective(objective_xQx , GRB.MINIMIZE)
        t.e("obj")
        t.s("optimize")
        m.optimize()
        t.e("optimize")
        if (m.status != GRB.OPTIMAL):
            print("optimal solution NOT found, status code=%d"%m.status)
        print("objective val = %.4f"%m.objVal)
        t.e()
        return (x.x,u.x)




