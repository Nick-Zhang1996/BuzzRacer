import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np

if (len(sys.argv) != 2):
    print("please specify a filename")
else:
    filename = sys.argv[1]

with open(filename, 'rb') as f:
    data = pickle.load(f)

# data dimension: [time_steps, cars, entries]
# for example data[100,3,:] denote the state at 1.0s(100th time step), for car no.4
# the states are listed below
data = np.array(data)[:,0,:]

t = data[:,0]
t = t-t[0]
x = data[:,1]
y = data[:,2]
heading = data[:,3]
v_forward = data[:,4]
v_sideway = data[:,5]
# angular speed
omega = data[:,6]
steering = data[:,7]
throttle = data[:,8]

plt.plot(x,y)
plt.show()
