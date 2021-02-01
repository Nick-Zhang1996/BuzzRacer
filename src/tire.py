# visually test tire curve
import matplotlib.pyplot as plt
import numpy as np
from math import radians,degrees


#get lateral acceleration from slip angle (rad
def oldTireCurve(slip):
    # satuation slip angle
    Bf = 10.0
    # peel-away sharpness, lower Cf = more gental peel away(street tire)
    Cf = 0.1
    # no slip tire stiffness
    # use this to control 1deg = 0.1g
    Df = 1.0*(180.0/np.pi)/ Bf / Cf
    retval = Df * np.sin( Cf * np.arctan( Bf * slip ) )
    return retval

# slip: slip angle in rad
# output: lateral friction coefficient
def newTireCurve(slip):
    slip = slip/np.pi*180.0
    # pacejktra needs deg
    B = 0.714
    C = 1.4
    D = 1.0
    E = -0.2
    retval = D * np.sin(C * np.arctan(B * slip - np.arctan(B*slip)))
    return retval

# slip: slip angle in rad
# output: lateral friction coefficient
def tireCurve(slip):
    slip = slip/np.pi*180.0
    # pacejktra needs deg
    B = 0.714
    C = 1.4
    D = 1.0
    E = -0.2
    retval = D * np.sin(C * np.arctan(B * slip))
    return retval

'''
xx = np.linspace(-10.0,10.0)
acc = np.arctan(xx)
plt.plot(xx, acc)
plt.show()
'''

if __name__=="__main__":

    xx = np.linspace(radians(-50),radians(50),1000)
    acc = tireCurve(xx)
    acc_alt = newTireCurve(xx)

    fig = plt.figure()
    ax = fig.gca()
    ax.plot(xx/np.pi*180.0, acc, label="original")
    ax.plot(xx/np.pi*180.0, acc_alt,label="new")
    ax.legend()
    plt.show()

    '''
    xx = np.linspace(-20,20)
    B = 0.714
    C = 1.4
    D = 1.0
    E = -0.2
    y = B * xx - np.arctan(B*xx)
    y2 = B*xx
    plt.plot(xx,y)
    plt.plot(xx,y2)
    plt.show()

    '''
