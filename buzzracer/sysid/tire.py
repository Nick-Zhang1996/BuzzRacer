# visually test tire curve
import matplotlib.pyplot as plt
import numpy as np
from math import radians,degrees
# slip: slip angle in rad
# output: lateral friction coefficient
def oldoldTireCurve(slip):
    C = 2.80646
    B = 0.51943
    Df = 3.93731*1.5
    Dr = 6.23597
    retval = Df * np.sin( C * np.arctan(B *slip)) 
    return retval

# slip: slip angle in rad
# output: lateral friction coefficient
def oldtireCurve(slip):
    C = 1.3
    B = 12.0/3
    D = 0.66*3
    # C: tail shape
    retval = D * np.sin( C * np.arctan(B *slip)) 
    return retval

def tireCurve(slip):
    C = 1.6
    B = 2.3
    D = 1.1
    # C: tail shape
    retval = D * np.sin( C * np.arctan(B *slip)) 
    return retval

'''
xx = np.linspace(-10.0,10.0)
acc = np.arctan(xx)
plt.plot(xx, acc)
plt.show()
'''

if __name__=="__main__":

    xx = np.linspace(radians(-40),radians(40),1000)
    acc = tireCurve(xx)
    acc_old = oldtireCurve(xx)

    fig = plt.figure()
    ax = fig.gca()
    ax.plot(xx/np.pi*180.0, acc_old, label="original")
    ax.plot(xx/np.pi*180.0, acc,label="new")
    ax.plot([0,9.26,7.75,degrees(0.094),degrees(0.044)],[0,0.58,0.455,0.367,0.093],'*')
    ax.plot([-20,20],[1.1,1.1],'--')
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
