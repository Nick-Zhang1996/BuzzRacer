# Example: load a config, modify as needed, then save xml
# to run batch experiments, you should write a file like this to generate all configs needed, then call 
# python batchExperiment.py folder_of_config
from common import *
from xml.dom import minidom
import xml.etree.ElementTree as ET
import numpy as np
from copy import deepcopy
import sys
from math import radians
from track import TrackFactory
from scipy.interpolate import splprep, splev,CubicSpline,interp1d
from extension.simulator.CurvilinearSimulator import CurvilinearSimulator

if (len(sys.argv) == 2):
    name = sys.argv[1]
else:
    print_error("you must specify a folder name under configs/")


config_folder = './configs/' + name + '/'
config_filename = config_folder + 'master.xml'
original_config = minidom.parse(config_filename)

config_track= original_config.getElementsByTagName('track')[0]
track = TrackFactory.build(main=None,config=config_track)
track.init()

class FakeMain():
    def __init__(self):
        self.extensions = []
        self.track = None
sim = CurvilinearSimulator(FakeMain())
sim.track = track

def getRandomInitialStatePair():
    s0 = np.random.uniform(0.4,track.raceline_len_m-0.4)
    s1 = s0 + np.random.uniform(-0.35,0.35)
    n0 = np.random.uniform(-0.2,0.2)
    n1 = np.random.uniform(-0.2,0.2)
    drr0 = splev(s0,track.raceline_s,der=1)
    '''
    heading0 = np.arctan2(drr0[1],drr0[0])
    drr1 = splev(s1,track.raceline_s,der=1)
    heading1 = np.arctan2(drr0[1],drr0[0])
    '''
    v0 = np.random.uniform(3.0,4.0)
    v1 = np.random.uniform(3.0,4.0)
    x0 = np.array((s0,v0,n0,0))
    x1 = np.array((s1,v1,n1,0))
    delta_x = x0 - x1
    is_in_collision = np.abs(delta_x[0])<0.18 and np.abs(delta_x[2])<0.14
    if (is_in_collision):
        return getRandomInitialStatePair()
    else:
        s0 = sim.curv2Cart(x0)
        s1 = sim.curv2Cart(x1)
        return tuple(s0[:4]),tuple(s1[:4])

index = 0
Qop = [0,2,-2]


for i in range(50):
    s0,s1 = getRandomInitialStatePair()
    for q0 in Qop:
        for q1 in Qop:
            config = deepcopy(original_config)
            config_extensions = config.getElementsByTagName('extensions')[0]
            config_cars = config.getElementsByTagName('cars')[0]
            config_car0 = config_cars.getElementsByTagName('car')[0]
            config_car1 = config_cars.getElementsByTagName('car')[1]
            config_controller = config_car0.getElementsByTagName('controller')[0]
            #attrs = config_controller.attributes.items()
            config_controller.attributes['Qop1'] =  str(q0)
            config_controller.attributes['Qop2'] =  str(q1)

            config_car0.getElementsByTagName('init_states')[0].childNodes[0].data = str(s0)
            config_car1.getElementsByTagName('init_states')[0].childNodes[0].data = str(s1)

            with open(config_folder+'exp%d.xml'%(index),'w') as f:
                config.writexml(f)
            index += 1

print('generated %d configs'%index)
