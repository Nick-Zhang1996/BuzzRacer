# MPC based trajectory planner
import matplotlib.pyplot as plt
import numpy as np
from MPC import MPC
from time import time

class Planner:
    def __init__(self):
        return

    # generate a path
    def genPath(self):
        # parameter
        k_vec = np.linspace(0,2*np.pi,1000)
        x_vec = k_vec
        y_vec = np.sin(k_vec)
        # n*2
        # curvilinear frame reference
        self.ref_path = np.vstack([x_vec,y_vec]).T
        # left positive, left negetive
        self.track_width = 0.6
        self.left_limit = np.ones_like(k_vec)*self.track_width/2
        self.right_limit = -np.ones_like(k_vec)*self.track_width/2
        self.dr,self.ddr = self.calcDerivative(self.ref_path)
        self.curvature = self.calcCurvature(self.dr,self.ddr)

        # ccw 90 deg
        A = np.array([[0,-1],[1,0]])
        tangent_dir = A @ self.dr.T
        self.left_boundary = (tangent_dir * self.left_limit).T + self.ref_path
        self.right_boundary = (tangent_dir * self.right_limit).T + self.ref_path

        self.calcArcLen(self.ref_path)

    def demo(self):
        self.genPath()

        x0 = [1,0.15,0,0]
        xref = [7,-0.15,0,0]

        (u_vec, state_traj) = self.solve(x0,xref)
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


    def solve(self,x0,xref):
        mpc = MPC()
        # p: prediction horizon
        # state x:[s,n,ds,dn]
        # u: [us=dds,un=ddn] second derivative of s
        self.n = n = 4
        self.m = m = 2
        self.p = p = horizon = 50
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
        du_max = 1
        u_max = 1.5
        mpc.convertLtiPlanner(A,B,P,Q,xref_vec,x0,p,du_max,u_max)
        # add track boundary constraints
        self.constructStateLimits(mpc)
        mpc.solve()
        dt = time()-t
        print(mpc.h.shape)
        print("freq = %.2fHz"%(1/dt))
        # state_traj in curvilinear frame
        state_traj = mpc.F @ mpc.u + mpc.Ex0
        state_traj = state_traj.reshape((p,n))

        '''
        plt.plot(state_traj)
        plt.show()
        u = np.array(mpc.u).reshape((-1,2))
        plt.plot(u)
        plt.show()
        '''
        return (mpc.u,state_traj)

    # construct the state limits
    def constructStateLimits(self,mpc):
        # create additional lines for Gx<h
        # track boundary limits
        p = self.p
        M = np.kron(np.eye(p),np.array([[0,1,0,0]]))
        G1 = M @ mpc.F
        N = np.ones((p,1))*self.track_width/2
        h1 = N - M @ mpc.Ex0

        G2 = -M @ mpc.F
        h2 = N + M @ mpc.Ex0

        #mpc.G = np.vstack([mpc.G,G1,G2])
        #mpc.h = np.vstack([mpc.h,h1,h2])
        # NOTEtesting
        G1 = G1[::10,:]
        h1 = h1[::10,:]
        G2 = G2[::10,:]
        h2 = h2[::10,:]
        mpc.G = np.vstack([G1,G2])
        mpc.h = np.vstack([h1,h2])
        return


    def calcDerivative(self,curve):
        # find first and second derivative
        dr = []
        ddr = []
        n = curve.shape[0]
        for i in range(1,n-1):
            rl = self.ref_path[i-1,:]
            r = self.ref_path[i,:]
            rr = self.ref_path[i+1,:]
            points = [rl, r, rr]
            ((al,a,ar),(bl,b,br)) = self.lagrangeDer(points)
            dr.append(al*rl+a*r+ar*rr)
            ddr.append(bl*rl+b*r+br*rr)
        dr = np.array(dr)
        ddr = np.array(ddr)
        dr = np.vstack([dr[0],dr,dr[-1]])
        ddr = np.vstack([ddr[0],ddr,ddr[-1]])
        return (dr,ddr)

    # right turn negative curvature
    def calcCurvature(self,dr_vec,ddr_vec):
        # ccw 90 deg
        A = np.array([[0,-1],[1,0]])
        curvature = np.dot( (A @ dr_vec.T).T, ddr_vec.T).flatten()
        return curvature

    # calculate segment cumulative length
    def calcArcLen(self,path):
        dr = np.vstack([np.zeros((1,2)),np.diff(path, axis=0)])
        ds = np.linalg.norm(dr,axis=1)
        self.ref_path_s = np.cumsum(ds)

    def stateTrajToCartesianTraj(self,traj):
        cartesian_traj = []
        A = np.array([[0,-1],[1,0]])
        for i in range(traj.shape[0]):
            idx = np.searchsorted(self.ref_path_s,traj[i,0],side='left')
            xy = self.ref_path[idx] + A @ self.dr[idx] * traj[i,1]
            cartesian_traj.append(xy)
        return np.array(cartesian_traj)

    def plotStateTraj(self,traj):
        traj = self.stateTrajToCartesianTraj(traj)
        plt.plot(traj[:,0],traj[:,1])
        return

    def plotTrack(self):
        # get left boundary
        plt.plot(self.ref_path[:,0],self.ref_path[:,1],'--')
        plt.plot(self.left_boundary[:,0],self.left_boundary[:,1],'-')
        plt.plot(self.right_boundary[:,0],self.right_boundary[:,1],'-')

    # given state x =[s,n,ds,dn]
    # plot a dot where the car should be
    def plotCar(self,x):
        # ccw 90 deg
        A = np.array([[0,-1],[1,0]])
        idx = np.searchsorted(self.ref_path_s,x[0],side='left')
        car_pos = self.ref_path[idx] + A @ self.dr[idx] * x[1]
        plt.plot(car_pos[0],car_pos[1],'o')


    # given three points, calculate first and second derivative as a linear combination of the three points rl, r, rr, which stand for r_(k-1), r_k, r_(k+1)
    # return: 2*3, tuple
    #       ((al, a, ar),
    #        (bl, b, br))
    # where f'@r = al*rl + a*r + ar*rr
    # where f''@r = bl*rl + b*r + br*rr
    # ds, arc length between rl,r and r, rr 
    # if not specified, |r-rl|_2 will be used as approximation
    def lagrangeDer(self,points,ds=None):
        rl,r,rr = points
        dist = lambda x,y:((x[0]-y[0])**2 + (x[1]-y[1])**2)**0.5
        if ds is None:
            sl = -dist(rl,r)
            sr = dist(r,rr)
        else:
            sl = -ds[0]
            sr = ds[1]

        try:
            al = - sr/sl/(sl-sr)
            a = -(sl+sr)/sl/sr
            ar = -sl/sr/(sr-sl)

            bl = 2/sl/(sl-sr)
            b = 2/sl/sr
            br = 2/sr/(sr-sl)
        except Warning as e:
            print(e)

        return ((al,a,ar),(bl,b,br))
if __name__=='__main__':
    planner = Planner()
    planner.demo()
