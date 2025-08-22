import numpy as np
import matplotlib.pyplot as plt


def plotVector(base, vec):
    plt.plot([base[0], base[0]+vec[0]], [base[1], base[1]+vec[1]], '-*')
    return


aym = 5.0
axm = 10.0

ay = 7
ax = 9

tt = np.linspace(0, np.pi*2, 1000)
yy = aym*np.cos(tt)
xx = axm*np.sin(tt)
plt.plot(yy, xx)

plt.plot([0, ay], [0, ax])

theta = np.arctan2(ax/axm, ay/aym)
p = [aym*np.cos(theta), axm*np.sin(theta)]
et = np.array([ax/(axm**2), -ay/(aym**2)])
et = et/np.linalg.norm(et)
plotVector(p, et*5)

xx = np.linspace(0, axm)
J = 2
C = -(p[1] * ax/axm**2 + p[0]*ay/aym**2)

yy_plus = (J**0.5 - C - xx*ax/axm**2)/(ay/aym**2)
yy_neg = (-J**0.5 - C - xx*ax/axm**2)/(ay/aym**2)
plt.plot(yy_plus, xx)
plt.plot(yy_neg, xx)

R1 = 2*np.array([[ay**2/aym**4, ax*ay/(axm**2*aym**2)],
                [ax*ay/(axm**2*aym**2), ax**2/axm**4]])
r1_x = C*np.array([[2*ay/aym**2, 2*ax/axm**2]])
i = 10
u = np.array([[yy_plus[i], xx[i]]]).T
J = 0.5*u.T @ R1 @ u + r1_x @ u + C**2
print(J)

plt.axis('equal')
plt.show()
