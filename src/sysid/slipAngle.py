# study lateral acceleration vs slip angle
import pickle
import matplotlib.pyplot as plt
import numpy as np

def loadLog(filename=None):
    if (len(sys.argv) != 2):
        if (filename is None):
            print_error("Specify a log to load")
        print_info("using %s"%(filename))
    else:
        filename = sys.argv[1]
    with open(filename, 'rb') as f:
        log = pickle.load(f)
    log = np.array(log)
    log = log.squeeze(1)
    return log

def prepLog(log,skip=1):
    #time(),x,y,theta,v_forward,v_sideway,omega, car.steering,car.throttle
    t = log[skip:,0]
    t = t-t[0]
    x = log[skip:,1]
    y = log[skip:,2]
    heading = log[skip:,3]
    v_forward = log[skip:,4]
    v_sideway = log[skip:,5]
    omega = log[skip:,6]
    steering = log[skip:,7]
    throttle = log[skip:,8]

    Log = namedtuple('Log', 't x y heading v_forward v_sideway omega steering throttle')
    mylog = Log(t,x,y,heading,v_forward,v_sideway,omega,steering,throttle)
    return mylog

def plotRelation():
    # load log
    filename = "../../log/jan12/full_state1.p"
    rawlog = loadLog(filename)
    log = prepLog(rawlog,skip=1)
    dt = 0.01
    vx = np.hstack([0,np.diff(log.x)])/dt
    vy = np.hstack([0,np.diff(log.y)])/dt
    data_len = log.t.shape[0]

