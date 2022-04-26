import numpy as np
from time import time
import cvxopt
import matplotlib.pyplot as plt

class MPC:
    def __init__(self,):
        cvxopt.solvers.options['show_progress'] = False
        return

    # setup parameters
    def setup(self,n,m,l,p):
        # dim of state
        self.n = n
        # dim of action
        self.m = m
        # dim of output y
        self.l = l
        # prediction horizon
        self.p = p
        # last applied control command, used for smoothing constrain
        self.last_applied_u = [0]
        # last solution, used for initial guess in next step
        # this is fuzzy as time is not properly shifter
        self.last_u = None
        return

    # convert linear time variant model of following form to general form for cvxopt.qp()
    #  u* = argmin_u (y_k - y_ref_k).T @ P @ (y_k - y_ref_k) + u_k.T Q u_k (summed for all k=1..p)
    #  where  y_k = C @ x_k
    #  s.t. x_k+1 = Ak xk + Bk uk  NOTE diecrete system, do your own discretization
    # k = 1..p, prediction horizon for MPC

    # target form 
    # J = 1/2 x.T P x + q.T x, where x is u in pervious eq
    # Gx <= h
    # Ax = b
    # the result will be stored internally
    # user may call solve() to obtain solution

    # input:
    # A_vec: [A0, A1, Ak... Ap-1] or np.array of shape (p,n,n)
    # B_vec: [B0, B1, Bk... Bp-1], or np.array of shape (p,m,n)
    # where n is dim of states, m is dim of ctrl
    # C: y = Cx, output matrix, dim (l,n)
    # P: l*l, cost for y = Cx
    # Q: m*m, cost for u(control), 
    # P,Q must be symmetric, this is applied equally to all projected timesteps
    # y_ref : [y_ref_1, y_ref_2, ... ], or np.array of shape (p,l,1) reference output
    # x0 : initial/current state
    # p : prediction horizon/steps
    # du_max : (m,) max|uk - uk-1|, for each u channel
    # u_max  : (m,) max|uk|, for each u channel
    def convertLtv(self,A_vec,B_vec,C,P,Q,y_ref,x0,du_max,u_max):
        p = self.p
        n = self.n
        m = self.m
        l = self.l
        A = list(A_vec)
        B = list(B_vec)
        assert len(A_vec) == p
        assert A[0].shape == (n,n)
        assert len(B_vec) == p
        assert B[0].shape == (n,m)
        assert P.shape == (l,l)
        assert Q.shape == (m,m)
        assert y_ref.shape == (p,l,1) or y_ref.shape == (p,l)
        # vertically stack all ref states
        y_ref = y_ref.reshape([l*p,1])
        assert x0.shape == (n,)

        # expand to proper dimension
        # TODO absorb this to setup for performance
        P = np.kron(np.eye(p,dtype=int),P)
        Q = np.kron(np.eye(p,dtype=int),Q)

        # assemble cost function
        F = np.zeros([p*n,p*m])
        for i in range(p-1):
            F[i*n:(i+1)*n,i*m:(i+1)*m] = B[i]
            F[(i+1)*n:(i+2)*n,:] = A[i] @ F[i*n:(i+1)*n,:]

        F[(p-1)*n:,(p-1)*m:] = B[p-1]

        E = np.empty([n*p,1])
        E[0*n:1*n] = x0.reshape(-1,1)
        for i in range(1,p):
            E[i*n:(i+1)*n] = A[i] @ E[(i-1)*n:i*n]

        G = np.kron(np.eye(p,dtype=int),C)

        P_qp = F.T @ G.T @ P @ G @ F + Q
        q_qp = F.T @ G.T @ P @ G @ E - F.T @ G.T @ P @ y_ref

        # assemble constrains
        # u_k+1 - uk
        G1 = np.hstack([-np.eye((p-1)*m),np.zeros([(p-1)*m,m])]) \
                + np.hstack([np.zeros([(p-1)*m,m]),np.eye((p-1)*m)])
        du_max = du_max.flatten()
        h1 = np.hstack([du_max]*(p-1))

        G2 = np.eye(p*m)
        h2 = np.hstack([u_max]*p)
        
        G3 = - np.eye(p*m)
        h3 = np.hstack([u_max]*p)

        # give special treatment to u0
        # du constrain on u0 is calculated from u from last time step
        # |u0 - last_applied_u| < du_max
        G4 = np.zeros([m,p*m])
        G4 = np.block([np.eye(m), np.zeros([m,(p-1)*m])])
        h4 = np.array(du_max).flatten() + np.array(self.last_applied_u).flatten()
        #print(self.last_applied_u)

        G5 = np.zeros([m,p*m])
        G5 = np.block([-np.eye(m), np.zeros([m,(p-1)*m])])
        h5 = np.array(du_max).flatten() - np.array(self.last_applied_u).flatten()
        #print(h5)
        #print('--')

        G_qp = np.vstack([G1,G2,G3,G4,G5])
        # TODO verify dimension
        h_qp = np.hstack([h1,h2,h3,h4,h5]).T

        # DEBUG
        self.E = E
        self.F = F

        self.P = P_qp
        self.q = q_qp
        self.G = G_qp
        self.h = h_qp
        return

    # convert a linear time invariant problem of form 
    #  u* = argmin_u (x-x_ref).T P (x-x_ref) + u.T Q u
    #  s.t. x_k+1 = A xk + B uk 
    # k = 0..h
    # A = R(n*p)(n*p)
    # B = R(n*p)(m*p)
    # to form 
    # J = 1/2 x.T P x + q.T x, where x is u in pervious eq
    # Gx <= h
    # Ax = b
    # the result will be stored internally
    # user may call solve() to obtain solution
    # inputs:
    # A = Rnn
    # B = Rnm
    # xref = p,n,1 -> reference state / target state
    # p horizon
    # du_max max|uk - uk-1|
    # u_max max|uk|
    # P is for a single state vector, it will be broadcasted automatically
    # NOTE currently for single input only
    def convertLti(self,A,B,P,Q,xref,x0,p,du_max,u_max):
        p = self.p
        n = self.n
        m = self.m
        l = self.l
        assert A.shape == (n,n)
        assert B.shape == (n,m)
        assert P.shape == (l,l)
        assert Q.shape == (m,m)
        assert x0.shape == (n,1)
        assert xref.shape == (p,n,1)

        P = np.kron(np.eye(p,dtype=int),P)
        Q = np.kron(np.eye(p,dtype=int),Q)

        # TODO record these in setup
        n = A.shape
        n = n[1]

        m = B.shape
        m = m[1]


        # assemble cost function
        F = np.zeros([p*n,p*m])
        for i in range(p-1):
            F[i*n:(i+1)*n,i*m:(i+1)*m] = B
            F[(i+1)*n:(i+2)*n,:] = A @ F[i*n:(i+1)*n,:]

        F[(p-1)*n:,(p-1)*m:] = B

        E = np.empty([n*p,1])

        E[0*n:1*n] = x0
        for i in range(1,p):
            E[i*n:(i+1)*n] = A @ E[(i-1)*n:i*n]

        P_qp = F.T @ P @ F + Q
        q_qp = F.T @ P @ E - F.T @ P @ xref.reshape([n*p,1])

        # assemble constrains
        # u_k+1 - uk
        # NOTE this can only handle 1d
        G1 = np.hstack([-np.eye(p*m-1),np.zeros([p*m-1,1])]) \
                + np.hstack([np.zeros([p*m-1,1]),np.eye(p*m-1)])
        h1 = np.ones([m*p-1,1]) * du_max

        G2 = np.eye(p*m)
        h2 = np.ones([m*p,1]) * u_max
        
        G3 = - np.eye(p*m)
        h3 = np.ones([m*p,1]) * u_max

        G_qp = np.vstack([G1,G2,G3])
        h_qp = np.vstack([h1,h2,h3])

        self.E = E
        self.F = F

        self.P = P_qp
        self.q = q_qp
        self.G = G_qp
        self.h = h_qp
        return

    # convert a linear time invariant problem of form 
    #  u* = argmin_u (x-x_ref).T P (x-x_ref) + u.T Q u
    #  s.t. x_k+1 = A xk + B uk 
    # k = 0..h
    # A = R(n*p)(n*p)
    # B = R(n*p)(m*p)
    # to form 
    # J = 1/2 x.T P x + q.T x, where x is u in pervious eq
    # Gx <= h
    # Ax = b
    # the result will be stored internally
    # user may call solve() to obtain solution
    # inputs:
    # A = Rnn
    # B = Rnm
    # xref = p,n,1 -> reference state / target state
    # p horizon
    # du_max max|uk - uk-1|
    # u_max max|uk|
    # P is for a single state vector, it will be broadcasted automatically
    # NOTE currently for single input only
    def convertLtiPlanner(self,A,B,P,Q,xref,x0,p,u_max=None,du_max=None):
        p = self.p
        n = self.n
        m = self.m
        l = self.l
        assert A.shape == (n,n)
        assert B.shape == (n,m)
        assert P.shape == (l,l)
        assert Q.shape == (m,m)
        assert x0.shape == (n,1)
        assert xref.shape == (p,n,1)

        P = np.kron(np.eye(p,dtype=int),P)
        Q = np.kron(np.eye(p,dtype=int),Q)

        # TODO record these in setup
        n = A.shape
        n = n[1]

        m = B.shape
        m = m[1]


        # assemble cost function
        F = np.zeros([p*n,p*m])
        for i in range(p-1):
            F[i*n:(i+1)*n,i*m:(i+1)*m] = B
            F[(i+1)*n:(i+2)*n,:] = A @ F[i*n:(i+1)*n,:]
        F[(p-1)*n:,(p-1)*m:] = B


        # E @ x0
        Ex0 = np.empty([n*p,1])
        Ex0[0*n:1*n] = A @ x0
        for i in range(1,p):
            Ex0[i*n:(i+1)*n] = A @ Ex0[(i-1)*n:i*n]

        P_qp = F.T @ P @ F + Q
        q_qp = F.T @ P @ Ex0 - F.T @ P @ xref.reshape([n*p,1])

        # assemble constrains
        # u_k+1 - uk
        # NOTE this can only handle 1d
        G_vec = []
        h_vec = []

        if (not du_max is None):
            G0 = np.hstack([-np.eye((p-1)*m),np.zeros([(p-1)*m,m])]) \
                    + np.hstack([np.zeros([(p-1)*m,m]),np.eye((p-1)*m)])
            h0 = np.kron(np.ones([(p-1),1]),du_max)

            #G1 = np.hstack([np.eye((p-1)*m),np.zeros([(p-1)*m,m])]) \
            #        + np.hstack([np.zeros([(p-1)*m,m]),-np.eye((p-1)*m)])
            #h1 = np.ones([(p-1),1]) * du_max
            G1 = -G0
            h1 = h0
            G_vec.append(G0)
            G_vec.append(G1)
            h_vec.append(h0)
            h_vec.append(h1)

        if (not u_max is None):
            # u_max
            G2 = np.eye(p*m)
            h2 = np.kron(np.ones([p,1]),u_max)
            G3 = - np.eye(p*m)
            h3 = np.kron(np.ones([p,1]),u_max)
            G_vec.append(G2)
            G_vec.append(G3)
            h_vec.append(h2)
            h_vec.append(h3)

        if (len(G_vec)>0):
            G_qp = np.vstack(G_vec)
            h_qp = np.vstack(h_vec)
        else:
            G_qp = None
            h_qp = None

        self.Ex0 = Ex0
        self.F = F

        self.P = P_qp
        self.q = q_qp
        self.G = G_qp
        self.h = h_qp
        return

    # for debug interest, plot the expected trajectory if u is faithfully followed
    # call after solve()
    def debug(self):
        # x = F @ u + E
        self.u[1::2,0] = self.u[1,0]
        #print(self.u[1,0])
        new_u = np.ones_like(self.u) 
        x_pro = self.F @ self.u + self.E
        # retrieve e_cross, e_heading
        e_cross = x_pro[0::self.n]
        e_heading = x_pro[2::self.n]
        return 

    def solve(self):
        P_qp = cvxopt.matrix(self.P)
        q_qp = cvxopt.matrix(self.q)

        '''
        if (self.last_u is None):
            guess = None
        else:
            guess = {'x':cvxopt.matrix(self.last_u)}
        '''
        # not much difference
        guess = None
        #https://cvxopt.org/userguide/coneprog.html#cvxopt.solvers.conelp
        if (self.G is None):
            sol=cvxopt.solvers.qp(P_qp,q_qp)
        else:
            G = cvxopt.matrix(self.G)
            h = cvxopt.matrix(self.h)
            #sol=cvxopt.solvers.qp(P_qp,q_qp,G,h,initval=guess)
            sol=cvxopt.solvers.qp(P_qp,q_qp,G,h)
        #print(sol['status'])
        # DEBUG
        self.u = np.array(sol['x']) 
        self.last_applied_u =  np.array(sol['x'])[0,:]
        self.last_u = sol['x']
        print("solver status: ",sol['status'])
        return self.last_applied_u

if __name__ == "__main__":
    mpc = MPC()
