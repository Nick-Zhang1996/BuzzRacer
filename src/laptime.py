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
def getLaptime(ctrl_offset,track_obj,start_grid,start_dir,start_seqno):
    global saveGif,gifimages,img_track,count
    count += 1
    laptime = track_obj.initRaceline(start_grid,start_dir,start_seqno,offset=ctrl_offset)
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
    # initialize instance
    mk103 = RCPtrack()

    # define track

    # Reduced "L" Track
    #descrip = 'uuruurddddll'
    #track_size = (5,3)
    # add manual offset for each control points
    #adjustment = [0,0,0,0,0,0,0,0,0,0,0,0]
    #adjustment[4] = -0.5
    #adjustment[8] = -0.5
    #adjustment[9] = 0
    #adjustment[10] = -0.5
    #adjustment = np.array(adjustment)
    #start_grid = (2,2)
    #start_dir = 'd'
    #start_seqno = 4

    # Full Track
    descrip = 'uuurrullurrrdddddluulddl'
    track_size = (6,4)
    # add manual offset for each control points
    adjustment = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

    adjustment[0] = -0.2
    adjustment[1] = -0.2
    #bottom right turn
    adjustment[2] = -0.2
    adjustment[3] = 0.5
    adjustment[4] = -0.2

    #bottom middle turn
    adjustment[6] = -0.2

    #bottom left turn
    adjustment[9] = -0.2

    # left L turn
    adjustment[12] = 0.5
    adjustment[13] = 0.5

    adjustment[15] = -0.5
    adjustment[16] = 0.5
    adjustment[18] = 0.5

    adjustment[21] = 0.35
    adjustment[22] = 0.35

    # initialize track
    track_len = len(descrip)
    mk103.initTrack(descrip,track_size,scale=0.565)
    start_grid = (3,3)
    start_dir = 'd'
    start_seqno = 10

    img_track = mk103.drawTrack()

    print("benchmark laptime = "+str(getLaptime(adjustment,mk103,start_grid,start_dir,start_seqno)))

    fun = lambda x: x
    cons = tuple([{'type': 'ineq', 'fun': lambda x:x[i]} for i in range(track_len)])

    '''
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
    '''
    #bnds = ((-1,1),(-1,1),(-1,1),(-1,1),(-1,1),(-1,1),(-1,1),(-1,1),(-1,1),(-1,1),(-1,1),(-1,1))
    max_offset = 0.5
    bnds = tuple([(-max_offset,max_offset) for i in range(track_len)])
    #bnds = ((-max_offset,max_offset),(-max_offset,max_offset),(-max_offset,max_offset),(-max_offset,max_offset),(-max_offset,max_offset),(-max_offset,max_offset),(-max_offset,max_offset),(-max_offset,max_offset),(-max_offset,max_offset),(-max_offset,max_offset),(-max_offset,max_offset),(-max_offset,max_offset))

    #res = minimize(getLaptime,adjustment,args=(mk103),method='COBYLA',bounds=bnds,constraints=cons)
    res = minimize(getLaptime,adjustment,args=(mk103,start_grid,start_dir,start_seqno),method='SLSQP',bounds=bnds,constraints=cons)
    print(res)
    adjustment = res.x
    print(res.x)
    print("iter = %d"%count)
    print("gif len = %d"%len(gifimages))

    if saveGif:
        gifimages[0].save(fp="./optimization.gif",format='GIF',append_images=gifimages,save_all=True,duration = 100,loop=0)

    #plt.plot(laptime_vec)
    #plt.show()
