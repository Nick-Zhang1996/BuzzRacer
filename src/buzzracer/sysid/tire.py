"""Pacejka tire models with parameters from system identification experiments."""

import casadi as ca
import torch
import numpy as np

def tire_curve(slip, use_torch=False):
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
    if use_torch:
        retval = D * torch.sin(C * torch.arctan(B * slip))
    else:
        retval = D * np.sin(C * np.arctan(B * slip))
    return retval


def tire_curve_casadi(slip):
    """CasADi-compatible tire curve."""
    C = 1.6
    B = 2.3
    D = 1.1
    return D * ca.sin(C * ca.atan(B * slip))
