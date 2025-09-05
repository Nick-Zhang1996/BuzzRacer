''' Pacejka tire models with parameters from system identification experiments'''
from math import radians, degrees
from deprecated import deprecated

import numpy as np

def tire_curve(slip):
    ''' Tire curve
    Args:
        slip: slip angle in rad
    Returns:
        output: lateral friction coefficient.
         output*slip_angle*normal_force = lateral force for axle
    '''
    C = 1.6
    B = 2.3
    D = 1.1
    # C: tail shape
    retval = D * np.sin(C * np.arctan(B * slip))
    return retval

@deprecated
def oldold_tire_curve(slip):
    ''' Tire curve
    Args:
        slip: slip angle in rad
    Return:
        output: lateral friction coefficient
    '''
    C = 2.80646
    B = 0.51943
    Df = 3.93731*1.5
    Dr = 6.23597
    retval = Df * np.sin(C * np.arctan(B * slip))
    return retval

@deprecated
def oldtire_curve(slip):
    ''' Tire curve
    Args:
        slip: slip angle in rad
    Return:
        output: lateral friction coefficient
    '''
    C = 1.3
    B = 12.0/3
    D = 0.66*3
    # C: tail shape
    retval = D * np.sin(C * np.arctan(B * slip))
    return retval