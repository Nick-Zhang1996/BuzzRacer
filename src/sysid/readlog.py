import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np

if (len(sys.argv) != 2):
    #print("please specify a filename")
    filename = '../../log/2022_3_14_exp/full_state2.p'

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

print("control")
plt.plot(throttle)
plt.plot(steering)
plt.show()

print("trajectory")
plt.plot(x,y)
plt.show()

print("heading")
plt.plot(heading)
plt.plot(omega)
plt.show()

print("velocity")
plt.plot(v_forward)
plt.plot(v_sideway)
plt.show()

'''
ti = 1312
tf = 1502
mean_throttle = np.mean(throttle[ti:tf])
mean_v = np.mean(v_forward[ti:tf])
print(mean_v,mean_throttle)


'''
