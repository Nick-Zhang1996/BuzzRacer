from xml.dom import minidom
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0,'..')
sys.path.insert(0,'../..')
import car_racing_simulator.Track as Track
import numpy as np
import matplotlib.pyplot as plt
from track.TrackFactory import TrackFactory

class Test:
    def __init__(self):
        pass

    def loadOrcaTrack(self):
        config = json.load(open('config.json'))
        self.path = config['data_dir']
        self.N = config['n_track']
        # ref point location
        self.X = np.loadtxt(self.path + "x_center.txt")[:, 0]
        self.Y = np.loadtxt(self.path + "x_center.txt")[:, 1]
        # ??
        self.s = np.loadtxt(self.path + "s_center.txt")
        self.phi = np.loadtxt(self.path + "phi_center.txt")
        self.kappa = np.loadtxt(self.path + "kappa_center.txt")
        self.diff_s = np.mean(np.diff(self.s))

        self.d_upper = np.loadtxt(self.path + "con_inner.txt")
        self.d_lower = np.loadtxt(self.path + "con_outer.txt")
        self.d_upper[520:565] = np.clip(self.d_upper[520:565], 0.2, 1)
        # self.d_lower[520:565] = np.clip(self.d_lower[520:565], -1, -0.2)
        self.border_angle_upper = np.loadtxt(self.path + "con_angle_inner.txt")
        self.border_angle_lower = np.loadtxt(self.path + "con_angle_outer.txt")

    def loadRcpTrack(self):
        config = minidom.parse('config.xml')
        config_track= config.getElementsByTagName('track')[0]
        self.track = TrackFactory(config_track,'full')
        N,X,Y,s,phi,kappa,diff_s,d_upper,d_lower,border_angle_upper,border_angle_lower = self.track.getOrcaStyleTrack()

        self.N = N
        self.X = X
        self.Y = Y
        self.s = s
        self.phi = phi
        self.kappa = kappa
        self.diff_s = diff_s

        self.d_upper = d_upper
        self.d_lower = d_lower
        # not really used
        self.border_angle_upper = border_angle_upper
        self.border_angle_lower = border_angle_lower
        return



    def plot(self):
        print('x,y')
        plt.plot(self.X,self.Y)
        plt.plot(self.X[0],self.Y[0],'*')
        plt.plot(self.X[100],self.Y[100],'o')
        plt.show()

        print('s')
        plt.plot(self.s)
        plt.show()

        print('phi')
        plt.plot(self.phi)
        plt.show()

        print('kappa')
        plt.plot(self.kappa)
        plt.show()

        print('diff_s')
        plt.plot(self.diff_s)
        plt.show()

        plt.plot(self.d_upper,'--')
        plt.plot(self.d_lower)
        plt.show()

        plt.plot(self.border_angle_upper,'--')
        plt.plot(self.border_angle_lower)
        plt.plot(self.border_angle_upper-self.border_angle_lower,'r')
        plt.show()


# main
test = Test()
#test.loadRcpTrack()
test.loadOrcaTrack()
test.plot()
