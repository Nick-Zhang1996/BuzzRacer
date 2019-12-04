

class skidpad:
    def __init__(self):
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
    def localTrajectory(self,state):
        x = state[0]
        y = state[1]
        heading = state[2]
        vf = state[3]
        vs = state[4]
        omega = state[5]

        # find offset
        # negative offset means car is to the right of the trajectory(need to turn left)
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
