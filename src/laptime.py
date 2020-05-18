# calculate the adjustment vector to minimize laptime 
from RCPTrack import RCPtrack
from time import time
from scipy.optimize import minimize
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import sys

#make a gif of the optimization process
saveGif = True
gifimages = []
laptime_vec = []

# given control offset, get laptime
count = 0
def getLaptime(ctrl_offset,track_obj):
    global saveGif,gifimages,img_track,count
    count += 1
    laptime = track_obj.initRaceline((2,2),'d',4,offset=ctrl_offset)
    laptime_vec.append(laptime)
    sys.stdout.write('.')
    sys.stdout.flush()
    if saveGif:
        img_track_raceline = mk103.drawRaceline(img=img_track.copy())
        gifimages.append(Image.fromarray(cv2.cvtColor(img_track_raceline,cv2.COLOR_BGR2RGB)))
        #plt.imshow(img_track_raceline)
        #plt.show()
    return laptime


if __name__ == "__main__":
    # current track setup in mk103
    mk103 = RCPtrack()
    mk103.initTrack('uuruurddddll',(5,3),scale=0.565)
    img_track = mk103.drawTrack()

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
    print("iter = %d"%count)
    print("gif len = %d"%len(gifimages))

    if saveGif:
        gifimages[0].save(fp="./optimization.gif",format='GIF',append_images=gifimages,save_all=True,duration = 100,loop=0)

    #plt.plot(laptime_vec)
    #plt.show()
