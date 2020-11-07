import numpy as np
from time import time
import cvxopt
import matplotlib.pyplot as plt

class MPC:
    def __init__(self,):
        cvxopt.solvers.options['show_progress'] = False
        return

    # setup parameters
    def setup(self,n,m,p):
        # dim of state
        self.n = n
        # dim of action
        self.m = m
        # prediction horizon
        self.p = p
        return

    # convert linear time variant model of following form to general form for cvxopt.qp()
    #  u* = argmin_u (x_k-x_ref_k).T P (x_k-x_ref_k) + u_k.T Q u_k (summed for all k=1..p)
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
    # P: n*n, cost for x(state)
    # Q: m*m, cost for u(control), 
    # P,Q must be symmetric, this is applied equally to all projected states
    # x_ref : [x_ref_1, x_ref_2, ... ], or np.array of shape (p,n,1) reference/target state
    # x0 : initial/current state
    # p : prediction horizon/steps
    # du_max : (m,) max|uk - uk-1|, for each u channel
    # u_max  : (m,) max|uk|, for each u channel
    def convertLtv(self,A_vec,B_vec,P,Q,x_ref,x0,du_max,u_max):
        p = self.p
        n = self.n
        m = self.m
        A = list(A_vec)
        B = list(B_vec)
        assert len(A_vec) == p
        assert A[0].shape == (n,n)
        assert len(B_vec) == p
        assert B[0].shape == (n,m)
        assert P.shape == (n,n)
        assert Q.shape == (m,m)
        assert x_ref.shape == (p,n,1) or x_ref.shape == (p,n)
        # vertically stack all ref states
        x_ref = x_ref.reshape([n*p,1])
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

        P_qp = F.T @ P @ F + Q
        q_qp = F.T @ P @ E - F.T @ P @ x_ref

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

        G_qp = np.vstack([G1,G2,G3])
        # TODO verify dimension
        h_qp = np.hstack([h1,h2,h3]).T

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
        print(self.u[1,0])
        new_u = np.ones_like(self.u) 
        x_pro = self.F @ self.u + self.E
        # retrieve e_cross, e_heading
        e_cross = x_pro[0::self.n]
        e_heading = x_pro[2::self.n]
        return 

    def solve(self):
        P_qp = cvxopt.matrix(self.P)
        q_qp = cvxopt.matrix(self.q)
        G = cvxopt.matrix(self.G)
        h = cvxopt.matrix(self.h)
        sol=cvxopt.solvers.qp(P_qp,q_qp,G,h)
        #print(sol['status'])
        # DEBUG
        self.u = np.array(sol['x']) 
        return np.array(sol['x'])

if __name__ == "__main__":
    mpc = MPC()
