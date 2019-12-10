from Track import Track
from math import cos,sin,pi,atan2

class Skidpad(Track):
    def __init__(self):
        super(Track,self).__init__()
        return

    # initialize a skidpad, centered at origin
    def initSkidpad(self,radius, velocity, ccw=True):
        self.radius = radius
        self.velocity = velocity
        self.ccw = ccw
        return

    def ctrlCar(self,state,reverse=False):
        return

    #state: x,y,theta,vf,vs,omega
    # x,y referenced from skidpad frame
    def localTrajectory(self,state,ccw=True):
        x = state[0]
        y = state[1]
        heading = state[2]
        vf = state[3]
        vs = state[4]
        omega = state[5]

        # find offset
        # positive offset means car is to the left of the trajectory(need to turn right)
        r = (x**2+y**2)**0.5
        offset = r-self.radius
        if ccw:
            offset = - offset

        # find closest point on track
        phase = atan2(y,x)
        raceline_point = (self.radius*cos(phase),self.radius*sin(phase))

        # line orientation
        if ccw:
            raceline_orientation = phase+pi/2
            signed_curvature = 1.0/self.radius
        else:
            raceline_orientation = phase-pi/2
            signed_curvature = -1.0/self.radius

        # reference point on raceline,lateral offset, tangent line orientation, curvature(signed)
        return (raceline_point,offset,raceline_orientation,signed_curvature)
    # update car state with bicycle model, no slip
    # dt: time, in sec
    # v: velocity of rear wheel, in m/s
    # state: (x,y,theta), np array
    # return new state (x,y,theta)
# XXX directly copied from track.py
    def updateCar(self,state,throttle,steering,dt):
        # wheelbase, in meter
        # heading of pi/2, i.e. vehile central axis aligned with y axis,
        # means theta = 0 (the x axis of car and world frame is aligned)
        theta = state[2] - pi/2
        L = 98e-3
        dr = v*dt
        dtheta = dr*tan(beta)/L
        # specific to vehicle frame (x to right of rear axle, y to forward)
        if (beta==0):
            dx = 0
            dy = dr
        else:
            dx = - L/tan(beta)*(1-cos(dtheta))
            dy =  abs(L/tan(beta)*sin(dtheta))
        #print(dx,dy)
        # specific to world frame
        dX = dx*cos(theta)-dy*sin(theta)
        dY = dx*sin(theta)+dy*cos(theta)
# should be x,y,heading,vf,vs,omega
        return np.array([state[0]+dX,state[1]+dY,state[2]+dtheta,v,0,dtheta/dt])

if __name__ == "__main__":
    sp = Skidpad()
    sp.initSkidpad(radius=1,velocity=1)
    x = 0.0
    y = 1.0
    theta = pi/2
    vf = 1
    vs = 0
    omega = 0
    state = (x,y,theta,vf,vs,omega)
    print(sp.localTrajectory(state))
