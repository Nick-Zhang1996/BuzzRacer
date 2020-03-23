# calculate the adjustment vector to minimize laptime 
from track import RCPtrack
from time import time
from scipy.optimize import minimize
import numpy as np

# given control offset, get laptime
def getLaptime(ctrl_offset,track_obj):
    print('count')
    laptime = track_obj.initRaceline((2,2),'d',4,offset=ctrl_offset)
    return laptime


if __name__ == "__main__":
    # current track setup in mk103
    mk103 = RCPtrack()
    mk103.initTrack('uuruurddddll',(5,3),scale=0.565)
    # add manual offset for each control points
    adjustment = [0,0,0,0,0,0,0,0,0,0,0,0]
    adjustment[4] = -0.5
    adjustment[8] = -0.5
    adjustment[9] = 0
    adjustment[10] = -0.5
    adjustment = np.array(adjustment)
    print("benchmark laptime = "+str(getLaptime(adjustment,mk103)))

    fun = lambda x: x
    cons = ({'type': 'ineq', 'fun': lambda x:x[0]},
            {'type': 'ineq', 'fun': lambda x:x[1]},
            {'type': 'ineq', 'fun': lambda x:x[2]},
            {'type': 'ineq', 'fun': lambda x:x[3]},
            {'type': 'ineq', 'fun': lambda x:x[4]},
            {'type': 'ineq', 'fun': lambda x:x[5]},
            {'type': 'ineq', 'fun': lambda x:x[6]},
            {'type': 'ineq', 'fun': lambda x:x[7]},
            {'type': 'ineq', 'fun': lambda x:x[8]},
            {'type': 'ineq', 'fun': lambda x:x[9]},
            {'type': 'ineq', 'fun': lambda x:x[10]},
            {'type': 'ineq', 'fun': lambda x:x[11]})
    #bnds = ((-1,1),(-1,1),(-1,1),(-1,1),(-1,1),(-1,1),(-1,1),(-1,1),(-1,1),(-1,1),(-1,1),(-1,1))
    max_offset = 0.5
    bnds = ((-max_offset,max_offset),(-max_offset,max_offset),(-max_offset,max_offset),(-max_offset,max_offset),(-max_offset,max_offset),(-max_offset,max_offset),(-max_offset,max_offset),(-max_offset,max_offset),(-max_offset,max_offset),(-max_offset,max_offset),(-max_offset,max_offset),(-max_offset,max_offset))

    #res = minimize(getLaptime,adjustment,args=(mk103),method='COBYLA',bounds=bnds,constraints=cons)
    res = minimize(getLaptime,adjustment,args=(mk103),method='SLSQP',bounds=bnds,constraints=cons)
    print(res)
    adjustment = res.x
    print(res.x)
    mk103.initRaceline((2,2),'d',4,offset=adjustment)
