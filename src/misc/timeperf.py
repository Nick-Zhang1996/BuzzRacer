import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 15})
# time performance for iLQGame
# 10, 15, 20, 25, 30 steps
x = [10,15,20,25,30]
iter1 = [81.8,64.6,53.6,44.6,38.6]
iter2 = [55.9,41.3,31.9,26.2,22.4]
iter3 = [41.0,30.3,22.7,18.1,15.8]
iter4 = [32.7,23.7,18.27]

plt.plot(x,iter1,'o-',label='iter=1')
plt.plot(x,iter2,'o-',label='iter=2')
plt.plot(x,iter3,'o-',label='iter=3')
plt.plot(x[:3],iter4,'o-',label='iter=4')
plt.xlabel('Horizon(steps)')
plt.ylabel('Frequency(Hz)')
plt.legend()
plt.show()


# time performance for MPPI-IBR
iter1 = [210.8,174.1,145.8,119.9,104.1]
iter2 = [142.1,112.8,93.7,76.3,65.5]
iter3 = [105.8,82.9,67.5,56.0,47.7]
iter4 = [84.7,65.7,53.2,43.9,37.5]

plt.plot(x,iter1,'o-',label='iter=1')
plt.plot(x,iter2,'o-',label='iter=2')
plt.plot(x,iter3,'o-',label='iter=3')
plt.plot(x,iter4,'o-',label='iter=4')
plt.xlabel('Horizon(steps)')
plt.ylabel('Frequency(Hz)')
plt.legend()
plt.show()
