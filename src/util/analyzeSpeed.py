import pickle
import numpy as np
import matplotlib.pyplot as plt

index = 31
filename = f'../../log/2023_2_6_exp/full_state{index}.p'
with open(filename, 'rb') as f:
    data = pickle.load(f)
# time(),x,y,theta,v_forward,v_sideway,omega, car.steering,car.throttle
data = np.array(data).squeeze(1)
t = data[:,0]
x = data[:,1]
y = data[:,2]
heading = data[:,3]
vf = data[:,4]
vs = data[:,5]
omega = data[:,6]
steering = data[:,7]
throttle = data[:,8]

p = np.array([0.06246385,0.19171776])
throttle_diff = throttle - vf*p[0] - p[1]

filename = f'../../log/2023_2_6_exp/debug_dict{index}.p'
with open(filename, 'rb') as f:
    data = pickle.load(f)

target_v = data[0]['target_v']

v = np.sqrt(vs**2+vf**2)

plt.plot(target_v,'--')
plt.plot(vf,'-')
plt.plot(throttle_diff,'.-')
plt.show()
