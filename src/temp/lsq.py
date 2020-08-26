import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(0,10)
y = (1-9*x)/4000
x += (np.random.random(size=(50))-0.5)/100
y += (np.random.random(size=(50))-0.5)/100

plt.plot(x,y,'o',label='Original Data',markersize=10)

x = x.reshape(-1,1)
y = y.reshape(-1,1)

A = np.hstack([x,y])
b = np.ones((50,1))
a,b = np.linalg.lstsq(A,b)[0]
print(a,b)
slope = -a/b
offset = 1/b
xx = np.linspace(0,10)
yy = slope*xx + offset
plt.plot(xx,yy,'r',label="fitted line")

plt.show()

A = np.hstack([x,np.ones((50,1))])
b = y
a,b = np.linalg.lstsq(A,b)[0]
yy = a*xx+b

plt.plot(x,y,'o',label='Original Data',markersize=10)
plt.plot(xx,yy,'r',label="fitted line")
plt.show()
