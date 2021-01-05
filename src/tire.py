# visually test tire curve
import matplotlib.pyplot as plt
import numpy as np
from math import radians,degrees


#get lateral acceleration from slip angle (rad
def tireCurve(slip,Cf=0.1):
    # satuation slip angle
    Bf = 10.0
    # peel-away sharpness, lower Cf = more gental peel away(street tire)
    Cf = 0.1
    # no slip tire stiffness
    # use this to control 1deg = 0.1g
    Df = 1.0*(180.0/np.pi)/ Bf / Cf
    retval = Df * np.sin( Cf * np.arctan( Bf * slip ) )
    return retval

'''
xx = np.linspace(-10.0,10.0)
acc = np.arctan(xx)
plt.plot(xx, acc)
plt.show()
'''

xx = np.linspace(radians(-20),radians(20))
acc = tireCurve(xx)
acc01 = tireCurve(xx,0.1)
plt.plot(xx/np.pi*180.0, acc)
plt.show()

