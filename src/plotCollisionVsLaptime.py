import sys
from common import *
import pickle
import matplotlib.pyplot as plt

def plotFile(filename,marker,label):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
        data = data[0]

    laptime_vec = np.array(data['laptime_vec'])
    collision_vec = np.array(data['collision_vec'])

    laptime_vec = laptime_vec[1:]
    # logged collision is cumulative collision
    collision_vec = np.diff(collision_vec)

    mask = laptime_vec > 1.0
    laptime_vec = laptime_vec[mask]
    collision_vec = collision_vec[mask]

    valid_laps = np.sum(mask)
    mean_laptime = np.mean(laptime_vec)
    cov_laptime = np.cov(laptime_vec)
    mean_collision = np.sum(collision_vec)/float(valid_laps)
    print(label + "valid laps: %d, laptime(mean/cov): %.4f/%.4f per lap collision %.2f"%(valid_laps, mean_laptime, cov_laptime, mean_collision))
    
    plt.plot(laptime_vec, collision_vec, marker, label=label)


if __name__=='__main__':
    plotFile("../log/kinematics_results/debug_dict8.p",'+','MPPI (same injected noise)')
    plotFile("../log/kinematics_results/debug_dict9.p",'o','MPPI (same terminal covariance)')
    plotFile("../log/kinematics_results/debug_dict10.p",'x','CCMPPI-mixed')
    plotFile("../log/kinematics_results/debug_dict11.p",'*','CCMPPI-pure')
    plt.xlabel("Laptime (s)")
    plt.ylabel("Collision")
    plt.legend()
    plt.show()
