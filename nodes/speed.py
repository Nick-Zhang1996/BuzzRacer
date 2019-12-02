#!/usr/bin/python
import sys
from threading import Lock
from math import radians,degrees,isnan,cos,sin
import numpy as np
sys.path.insert(0,'../src')
from track import RCPtrack,TF
from vicon import Vicon
import pickle
import matplotlib.pyplot as plt
lock_state = Lock()
# (x,y,heading,v_longitudinal, v_lateral, angular rate)
local_state = None
previous_state = (0,0,0,0,0,0)
vicon_dt = 0.01

tf = TF()
vi = Vicon()

# track pose
q_t = tf.euler2q(radians(180),0,radians(90))
T = np.hstack([q_t,np.array([0,1,0])])

local_state = None
previous_state = (0,0,0,0,0,0)

def updateloop():
    global previous_state
    global local_state
    # state update
    (x,y,z,rx,ry,rz) = vi.getViconUpdate()
    # get body pose in track frame
    (x,y,heading) = tf.reframeR(T,x,y,z,tf.euler2Rxyz(rx,ry,rz))
    vx = (x - previous_state[0])/vicon_dt
    vy = (y - previous_state[1])/vicon_dt
    omega = (heading - previous_state[2])/vicon_dt
    vf = -vx*cos(heading) - vy*sin(heading)
    vs = vx*sin(heading) - vy*cos(heading)

    lock_state.acquire()
    local_state = (x,y,heading, vf, vs, omega)
    previous_state = local_state
    lock_state.release()

if __name__ == '__main__':
    max_speed = 0
    max_omega = 0
    max_lateral = 0
    updateloop()
    updateloop()
    speed_vec = []
    omega_vec = []
    slide_vec = []
    heading_vec = []

    for i in range(1000):
        updateloop()
        l = local_state
        #print("%6.3f %6.3f %6.3f %6.3f %6.3f %6.3f"%(l[0], l[1], l[2], l[3], l[4], l[5]))
        speed_vec.append(l[3])
        omega_vec.append(l[5])
        slide_vec.append(l[4])
        heading_vec.append(l[2])
        if(abs(l[3])>max_speed):
            max_speed = abs(l[3])
            print("max_speed: "+str(max_speed))
        if(abs(l[4])>max_lateral):
            max_lateral = abs(l[4])
            print("max_lateral: "+str(max_lateral))
        if(abs(l[5])>max_omega):
            max_omega = abs(l[5])
            print("max_omega: "+str(max_omega))

    '''
    speed_dump = open('speed.p','wb')
    pickle.dump(speed_vec,speed_dump)
    speed_dump.close()

    slide_dump = open('slide.p','wb')
    pickle.dump(slide_vec,slide_dump)
    slide_dump.close()

    omega_dump = open('omega.p','wb')
    pickle.dump(omega_vec,omega_dump)
    omega_dump.close()
    '''

    plt.plot(heading_vec)
    plt.show()
    heading_dump = open('heading.p','wb')
    pickle.dump(heading_vec,heading_dump)
    heading_dump.close()
    
    #plt.subplot(311)
    #plt.plot(speed_vec)
    #plt.subplot(312)
    #plt.plot(slide_vec)
    #plt.subplot(313)
    #plt.plot(omega_vec)
    #plt.show()
