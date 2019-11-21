#!/usr/bin/python
from track import RCPtrack,TF
from vicon import Vicon
lock_state = Lock()
# (x,y,heading,v_longitudinal, v_lateral, angular rate)
local_state = None
previous_state = (0,0,0,0,0,0)
vicon_dt = 0.02

tf = TF()
vi = Vicon()

def updateloop():
    # state update
    (x,y,z,rx,ry,rz) = vi.getViconUpdate()
    # get body pose in track frame
    (x,y,heading) = tf.reframeR(T,x,y,z,tf.euler2Rxyz(rx,ry,rz))
    vx = (x - previous_state[0])/vicon_dt
    vy = (y - previous_state[1])/vicon_dt
    omega = (omega - previous_state[2])/vicon_dt
    vf = vx*cos(heading) + vy*sin(heading)
    vs = vx*sin(heading) - vy*cos(heading)

    lock_state.acquire()
    local_state = (x,y,heading, vf, vs, omega)
    previous_state = local_state
    lock_state.release()

if __name__ == '__main__':
    while True:
        updateloop()
        print(local_state)

