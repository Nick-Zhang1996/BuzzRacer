import numpy as np
from time import time
import cvxopt

class MPC:
    def __init__(self,):
        cvxopt.solvers.options['show_progress'] = False


        return

    # convert problem of form 
    #  u* = argmin_u (x-x_ref).T P (x-x_ref) + u.T Q u
    #  s.t. x_k+1 = Ak xk + Bk uk 
    # k = 0..h
    # Ak = Rnn
    # Bk = Rnm
    def setup(self):
        return

    # linear time variant model, where Ak(t)
    # input:
    # A: [A1, A2, A... Ap] or np.matrix of shape p,n*p,n*m
    # B: [B1, B2, B... Bp], where p is horizon steps, n is dim of states, m is dim of ctrl
    # if n=1 this is automatically a LTI problem
    # convert a linear time variant problem of form 
    #  u* = argmin_u (x-x_ref).T P (x-x_ref) + u.T Q u
    #  s.t. x_k+1 = Ak xk + Bk uk 
    # k = 0..h
    # A = R(n*p)(n*p)
    # B = R(n*p)(m*p)
    # to form 
    # J = 1/2 x.T P x + q.T x, where x is u in pervious eq
    # Gx <= h
    # Ax = b
    # the result will be stored internally
    # user may call solve() to obtain solution
    def convertLtv(self,A,B):
        F = np.zeros(p,m)
        for i in range(p-1):
            F[i,i] = B[i]
            F[i+1,:] = A[i] @ F[i,:]
        F[p-1,p-1] = B[p-1]
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
        sol = self.solve()

        return

    def solve(self):
        P_qp = cvxopt.matrix(self.P)
        q_qp = cvxopt.matrix(self.q)
        G = cvxopt.matrix(self.G)
        h = cvxopt.matrix(self.h)
        sol=cvxopt.solvers.qp(P_qp,q_qp,G,h)
        #print(sol['status'])
        return sol['x']

