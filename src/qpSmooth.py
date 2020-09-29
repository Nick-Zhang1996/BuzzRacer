# smooth path with quadratic programming
# following paper "A quadratic programming approach to path smoothing"

from RCPTrack import RCPtrack
import numpy as np
import matplotlib.pyplot as plt
from common import *

class QpSmooth:
    # track: RCPtrack object
    def __init__(self,track):
        return


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

        al = - sr/sl/(sl-sr)
        a = -(sl+sr)/sl/sr
        ar = -sl/sr/(sr-sl)

        bl = 2/sl/(sl-sr)
        b = 2/sl/sr
        br = 2/sr/(sr-sl)

        return ((al,a,ar),(bl,b,br))

    # construct a fifth order bezier curve passing through endpoints r
    # matching first and second derivative dr, ddr
    # r (2,2)
    # dr (2,2),  taken w.r.t. arc length s
    # ddr (2,2), taken w.r.t. arc length s 
    # ds: arc length between the endpoints
    def bezierCurve(self,r,dr,ddr,ds=None):
        rl,rr = r
        drl,drr = dr
        ddrl,ddrr = ddr

        dist = lambda x,y:((x[0]-y[0])**2 + (x[1]-y[1])**2)**0.5
        if ds is None:
            ds = dist(rl,rr)

        # two sets of equations, one for x, one for y
        #bx = np.matrix([rl[0],rr[0],drl[0],drr[0],ddrl[0],ddrr[0]]).T
        #by = np.matrix([rl[1],rr[1],drl[1],drr[1],ddrl[1],ddrr[1]]).T

        # dr = dr/ds = dr/dt * dt/ds
        # we want dB/dt = dr/dt = dr(input) * ds/dt = dr * ds(between two endpoints)
        bx = np.matrix([rl[0],rr[0],drl[0]*ds,drr[0]*ds,ddrl[0]*ds*ds,ddrr[0]*ds*ds]).T
        by = np.matrix([rl[1],rr[1],drl[1]*ds,drr[1]*ds,ddrl[1]*ds*ds,ddrr[1]*ds*ds]).T
        b = np.hstack([bx,by])

        # x_x = P0_x, P1_x ... P5_x
        # x_y = P0_y, P1_y ... P5_y
        A = [[ 1, 0, 0, 0, 0, 0],
             [ 0, 0, 0, 0, 0, 1],
             [-5, 5, 0, 0, 0, 0],
             [ 0, 0, 0, 0,-5, 5],
             [20,-40,20,0, 0, 0],
             [0 , 0, 0,20,-40,20]]
        A = np.matrix(A)

        try:
            sol = np.linalg.solve(A,b)
        except LinAlgError:
            print_error("can't solve bezier Curve")

        # return the control points
        P = sol.T
        return P

    # generate a bezier spline matching derivative estimated from lagrange interpolation
    # return: vector function, domain [0,len(points)]
    def bezierSpline(self,ctrl_pnts):
        ctrl_pnts = np.array(ctrl_pnts)

        # calculate first and second derivative
        # w.r.t. ds, estimated with 2-norm
        df = []
        ddf = []
        n = points.shape[1]
        for i in range(n):
            rl = ctrl_pnts[:,(i-1)%n]
            r  = ctrl_pnts[:,(i)%n]
            rr = ctrl_pnts[:,(i+1)%n]] 
            points = [rl, r, rr]
            
            ((al,a,ar),(bl,b,br)) = self.lagrangeDer(points)
            df.append( al*rl + a*r + ar*rr)
            ddf.append(bl*rl + b*r + br*rr)

        # generate bezier spline segments
        P = self.bezierCurve([rl,r],[df,df],[ddf,ddf],ds=None)





    def testLagrangeDer(self):
        # generate three points
        points = np.array([[-1,-1],[2,2],[5,5]])
        rl,r,rr = points
        ((al,a,ar),(bl,b,br)) = self.lagrangeDer(points)
        df = al*rl + a*r + ar*rr
        ddf = bl*rl + b*r + br*rr
        print(df)
        print(ddf)
        x = points[:,0]
        y = points[:,1]
        plt.plot(x,y)
        plt.show()
        return

    # test bezier curve
    def testBezierCurve(self):
        # generate three points
        points = np.array([[-1,-1],[2,2],[3,0]])
        rl,r,rr = points
        ((al,a,ar),(bl,b,br)) = self.lagrangeDer(points)
        df = al*rl + a*r + ar*rr
        ddf = bl*rl + b*r + br*rr
        print("df, ddf at r_k")
        print(df)
        print(ddf)
        x = points[:,0]
        y = points[:,1]

        P = self.bezierCurve([rl,r],[df,df],[ddf,ddf],ds=None)
        print(P)

        B = lambda t,p: (1-t)**5*p[0] + 5*t*(1-t)**4*p[1] + 10*t**2*(1-t)**3*p[2] + 10*t**3*(1-t)**2*p[3] + 5*t**4*(1-t)*p[4] + t**5*p[5]

        u = np.linspace(0,1)
        xx = B(u,np.array(P[0,:]).flatten())
        yy = B(u,np.array(P[1,:]).flatten())


        plt.plot(np.array(P[0,:]).flatten(),np.array(P[1,:]).flatten(),'ro')
        plt.plot(x,y,'bo')
        plt.plot(xx,yy)

        plt.show()

        return



if __name__ == "__main__":
    qp = QpSmooth(None)
    qp.testBezierCurve()
