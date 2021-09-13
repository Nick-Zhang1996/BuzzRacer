# analyze csv log, plot peratio frontier
import numpy as np
import matplotlib.pyplot as plt
from TrackFactory import TrackFactory
from common import *
import pickle
import cv2
filename = "log.txt"
#filename = "sat_fine_grid.txt"
filename = "combined.txt"

track = TrackFactory(name='full')
def plotTraj(track, filename, img, color, text):
    global offset
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    data = np.array(data).squeeze(1)

    x = data[:,1]
    y = data[:,2]
    points = np.vstack([x,y]).T
    track.drawPolyline(points, img, lineColor=color,thickness=2)
    return img

    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    # org
    org = (20, 50+offset)
    # fontScale
    fontScale = 1
    # Line thickness of 2 px
    thickness = 2
    # Using cv2.putText() method
    img = cv2.putText(img, text, org, font,
                       fontScale, color, thickness, cv2.LINE_AA)
    offset += 30

    return img

mppi = []
ccmppi = []

mppi_total_count = 0
ccmppi_total_count = 0

mppi_bad_count = 0
ccmppi_bad_count = 0

# load blank canvas
with open("track_img.p", 'rb') as f:
    track_img = pickle.load(f)


with open(filename, 'r') as f:
    for line in f:
        if(line[0] == '#'):
            continue
        entry = line.split(',')

        if (entry[0].lstrip() == 'ccmppi'):
            ccmppi_total_count += 1
            if (entry[9].lstrip() != "False"):
                ccmppi_bad_count += 1
                continue
            entry = entry[:9] + entry[10:]
            ccmppi.append([float(val) for val in entry[1:]])

        if (entry[0].lstrip() == 'mppi-same-injected'):
            mppi_total_count += 1
            if (entry[9].lstrip() != "False"):
                mppi_bad_count += 1
                continue
            entry = entry[:9] + entry[10:]
            mppi.append([float(val) for val in entry[1:]])

# print success rate:
print("mppi: total %d, success %d, success rate %.1f%%"%(mppi_total_count, mppi_total_count-mppi_bad_count, (mppi_total_count - mppi_bad_count)/mppi_total_count*100))
print("ccmppi: total %d, success %d, success rate %.1f%%"%(ccmppi_total_count, ccmppi_total_count-ccmppi_bad_count, (ccmppi_total_count - ccmppi_bad_count)/ccmppi_total_count*100))

# algorithm, samples, car_total_laps, laptime_mean(s),  collision_count
ccmppi = np.array(ccmppi)
mppi = np.array(mppi)

a_low_thresh = 0.5
a_high_thresh = 0.9
b_low_thresh = 0
b_high_thresh = 6
a_low_thresh = 0
a_high_thresh = 10
b_low_thresh = 0
b_high_thresh = 60
mask1 = np.bitwise_and(ccmppi[:,8] > a_low_thresh, ccmppi[:,8] < a_high_thresh)
mask2 = np.bitwise_and(ccmppi[:,9] > b_low_thresh, ccmppi[:,9] < b_high_thresh)
mask = np.bitwise_and(mask1,mask2)
ccmppi = ccmppi[mask]

mask1 = np.bitwise_and(mppi[:,8] > a_low_thresh, mppi[:,8] < a_high_thresh)
mask2 = np.bitwise_and(mppi[:,9] > b_low_thresh, mppi[:,9] < b_high_thresh)
mask = np.bitwise_and(mask1,mask2)
mppi = mppi[mask] 

mppi_mean_laptime = np.mean(mppi[:,2])
ccmppi_mean_laptime = np.mean(ccmppi[:,2])
mppi_mean_collision = np.mean(mppi[:,3])
ccmppi_mean_collision = np.mean(ccmppi[:,3])
print("mppi  : laptime %.3f, collision %.2f "%(mppi_mean_laptime, mppi_mean_collision))
print("ccmppi: laptime %.3f, collision %.2f "%(ccmppi_mean_laptime, ccmppi_mean_collision))
mppi_mean_cov = np.mean(mppi[:,5])
ccmppi_mean_cov = np.mean(ccmppi[:,5])
print("cov: mppi: %.5f, ccmppi: %.5f"%(mppi_mean_cov, ccmppi_mean_cov))

# plot all data
plt.plot(ccmppi[:,3], ccmppi[:,2],'o',label='ccmppi')
plt.plot(mppi[:,3], mppi[:,2],'o', label= 'MPPI')

plt.xlabel("Number of collisions")
plt.ylabel("Laptime (s)")
plt.legend()
plt.show()

# circle same config
# NOTE good index_cc = 4
#for index in range(mppi.shape[0]):
for index in [4]:
    index_cc = -1
    index_mppi = index 

    alfa = mppi[index_mppi,8]
    beta = mppi[index_mppi,9]
    print("mppi cov index %d, alfa %.2f beta %.2f"%(index_mppi, alfa, beta))
    for i in range(ccmppi.shape[0]):
        if (np.isclose(alfa,ccmppi[i,8]) and np.isclose(beta,ccmppi[i,9])): 
            index_cc = i
    if (index_cc == -1):
        print_error("can't find ccmppi index")

    print("log index: ccmppi: %d,  mppi: %d"%(ccmppi[index_cc,7], mppi[index_mppi,7]))

    # plot frontier with circled settings
    plt.plot(ccmppi[:,3], ccmppi[:,2],'o',label='CCMPPI')
    plt.plot(mppi[:,3], mppi[:,2],'o', label= 'MPPI')

    plt.scatter(ccmppi[index_cc,3], ccmppi[index_cc,2],s=80,facecolor='none', edgecolor='r',label='same setting', zorder=10)
    plt.scatter(mppi[index_mppi,3], mppi[index_mppi,2],s=80,facecolor='none', edgecolor='r', zorder=10)
    plt.xlabel("Number of collisions")
    plt.ylabel("Laptime (s)")
    plt.legend()
    plt.show()
    ccmppi_logno = int(ccmppi[index_cc,7])
    mppi_logno = int(mppi[index_mppi,7])

    filename = "../log/kinematics_results/full_state"+str(ccmppi_logno)+".p"
    img1 = plotTraj(track,filename, track_img.copy(), (0,0,255), "CCMPPI")

    filename = "../log/kinematics_results/full_state"+str(mppi_logno)+".p"
    img2 = plotTraj(track,filename, track_img.copy(), (0,255,0), "MPPI")

    img = (img1/2 + img2/2)
    img = np.array(img,dtype=np.uint8)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig = plt.figure()
    plt.imshow(img)
    fig.savefig('out.png', bbox_inches='tight',transparent=True, pad_inches=0)
    plt.show()

