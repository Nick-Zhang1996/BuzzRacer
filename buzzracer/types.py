
''' Define types used througout the project'''
from typing import NamedTuple

class Control(NamedTuple):
    steering: float
    ''' Steering angle for an Ackermann steering vehicle in rad, left positive '''
    throttle: float
    ''' Throttle, positive forward, negative braking. Typical rance [-1,1]
     In some VehicleDynamics, this is mapped to acceleration directly, 
      but in general does not have direct physical meaning '''

class CartesianState(NamedTuple):
    x: float
    ''' X coordinate in global frame, unit: m'''
    y: float
    ''' Y coordinate in global frame, unit: m'''
    heading: float
    ''' Heading in radians from x axis, ccw positive'''
    v_forward: float
    ''' Longitudinal speed m/s w.r.t. vehicle centerline, forward positive '''
    v_sideway: float
    ''' Lateral speed m/s w.r.t. vehicle centerline, left positive '''
    omega: float


class CurvilinearState(NamedTuple):
    ''' A state in Frenet frame'''
    progress: float
    ''' Progress along reference curve'''
    lateral_err: float
    ''' Lateral deviation, left positive'''
    rel_heading: float
    ''' Relative heading w.r.t. reference curve'''
    v_forward: float
    ''' Longitudinal speed, forward positive '''
    v_sideway: float
    ''' Lateral speed, left positive '''
    rel_omega: float
    ''' Relative heading time rate w.r.t. reference curve, ccw positive'''
