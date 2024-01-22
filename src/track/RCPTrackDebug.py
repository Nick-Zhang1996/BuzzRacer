# rarely used or obsolete methods for RCPTrack
import numpy as np
import os.path
from numpy import isclose
import matplotlib.pyplot as plt
from math import atan2,radians,degrees,sin,cos,pi,tan,copysign,asin,acos,isnan
from scipy.interpolate import splprep, splev,CubicSpline,interp1d
from scipy.optimize import minimize_scalar,minimize,brentq
from scipy.integrate import solve_ivp
from time import sleep,time
import cv2
from PIL import Image
import pickle
from bisect import bisect

from common import *
from util.timeUtil import execution_timer
from track.RCPTrack import RCPTrack

class RCPTrackDebug(RCPTrack):
    def __init__(self,main=None,config=None):
        super().__init__(main,config)

