from common import *
from xml.dom import minidom
import xml.etree.ElementTree as ET
import numpy as np
from copy import deepcopy
import sys
from math import radians
from track import TrackFactory
from scipy.interpolate import splprep, splev, CubicSpline, interp1d
from buzzracer.extension.simulator.curvilinear_simulator import CurvilinearSimulator
import matplotlib.pyplot as plt

if (len(sys.argv) == 2):
    name = sys.argv[1]
name = 'batch_ilqgame'

config_folder = './configs/' + name + '/'
config_filename = config_folder + 'master.xml'
original_config = minidom.parse(config_filename)

config_track = original_config.getElementsByTagName('track')[0]
track = TrackFactory.build(main=None, config=config_track)
track.init()


class FakeMain():
    def __init__(self):
        self.extensions = []
        self.track = None


sim = CurvilinearSimulator(FakeMain())
sim.track = track


def get_coord(s0):
    x0 = np.array((s0, 0, 0, 0))
    cart0 = sim.curv2_cart(x0)
    return cart0[:2]


ss = np.linspace(0, sim.track.raceline_len_m, 1000)
coord = np.array([get_coord(s) for s in ss])

plt.plot(coord[:, 0], coord[:, 1], 'o')
plt.show()
