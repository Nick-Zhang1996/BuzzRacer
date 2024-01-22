# find total time of log
import glob
import pickle
import numpy as np
log_names =  glob.glob('./full_state*.p')

total_duration = 0
for filename in log_names:
    with open(filename, 'rb') as f:
        data = pickle.load(f)
        data = np.array(data)
        t = data[:,0]
        duration = t[-1]-t[0]
        total_duration += duration
        print("%s %.1fs"%(filename,duration))
print("total %.1fs (%.1f min)"%(total_duration,total_duration/60))

