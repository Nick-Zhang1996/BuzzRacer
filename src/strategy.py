# code for validating strategy controller
import numpy as np
from qpSmooth import QpSmooth
from track.RCPTrackDebug import RCPTrackDebug as RCPTrack
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev,CubicSpline,interp1d

if __name__ == "__main__":
    fulltrack = QpSmooth()
    fulltrack.load()

    # create 2 speed profile with similar laptime

    # agent i, faster in straights
    retval = fulltrack.generateSpeedProfile(
            mu=0.7, 
            acc_max_fun = lambda x:3.5,
            dec_max_fun = lambda x:3.5,
            )
    fulltrack.targetVfromU = speed_profile_fun_i = retval['speed_profile_fun']
    retval['max_v']
    retval['min_v']
    fulltrack.verifySpeedProfile(speed_profile_fun=speed_profile_fun_i)
    tt = []
    ss = []
    vv = []
    s = 0
    t = 0
    dt = 0.01

    # progress in time domain
    while (s<fulltrack.raceline_len_m):
        v = fulltrack.sToV(s)
        tt.append(t)
        ss.append(s)
        vv.append(v)
        s += v*dt
        t += dt
    print(f'total t = {tt[-1]}')
    plt.plot(ss,vv)
    # progress in s domain
    tt = []
    ss = []
    vv = []
    s = 0
    t = 0
    ds = 0.01
    while (s<fulltrack.raceline_len_m):
        tt.append(t)
        ss.append(s)
        vv.append(v)
        v = fulltrack.sToV(s)
        t += ds/v
        s += ds
    print(f'total t = {tt[-1]}')
    plt.plot(ss,vv)

    # progress in u domain
    tt = []
    ss = []
    s = 0
    t = 0
    n_steps = 1000
    xx = np.linspace(0,fulltrack.track_length_grid,n_steps+1)
    dist = lambda a,b: ((a[0]-b[0])**2+(a[1]-b[1])**2)**0.5
    #vv = speed_profile_fun_i(xx)
    vv = fulltrack.sToV(fulltrack.uToS(xx))
    for i in range(n_steps+1):
        tt.append(t)
        ss.append(s)
        (x_i, y_i) = splev(xx[i%n_steps], fulltrack.raceline, der=0)
        (x_i_1, y_i_1) = splev(xx[(i+1)%n_steps], fulltrack.raceline, der=0)
        # distance between two steps
        ds = dist((x_i, y_i),(x_i_1, y_i_1))
        s += ds
        t += ds/(vv[i%n_steps]+vv[(i+1)%n_steps])*2
    print(f'total t = {tt[-1]}')
    plt.plot(ss,vv,'*')

    plt.show()



    # agent j, faster in corners
    retval = fulltrack.generateSpeedProfile(
            mu=0.9, 
            acc_max_fun = lambda x:1.5,
            dec_max_fun = lambda x:1.5,
            )
    fulltrack.targetVfromU = speed_profile_fun_i = retval['speed_profile_fun']
    retval['max_v']
    retval['min_v']
    fulltrack.verifySpeedProfile(speed_profile_fun=speed_profile_fun_j)

    # visualize both cars
