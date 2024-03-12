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
from extension.simulator.KinematicBicycleFrenetSimulator import KinematicBicycleFrenetSimulator


class FakeMain():
    def __init__(self):
        self.extensions = []
        self.track = None


# leader = 0: s0 lead, =1: s1 lead, None: even
def getRandomInitialStatePair(leader=None,track=None):

    if (leader == 0):
        # s0 in front
        s1 = np.random.uniform(0.5,track.raceline_len_m-0.5)
        s0 = s1 + np.random.uniform(0.1,0.3)
        v0 = np.random.uniform(2.5,3.0)
        v1 = v0 + np.random.uniform(0.7,1.0)
    elif (leader == 1):
        # s1 in front
        s0 = np.random.uniform(0.5,track.raceline_len_m-0.5)
        s1 = s0 + np.random.uniform(0.1,0.3)
        v1 = np.random.uniform(2.5,3.0)
        v0 = v1 + np.random.uniform(0.7,1.0)
    else:
        s0 = np.random.uniform(0.5,track.raceline_len_m-0.5)
        s1 = s0 + np.random.uniform(0.3,0.5)
        v0 = np.random.uniform(2.5,3.0)
        v1 = np.random.uniform(2.5,3.0)

    n0 = np.random.uniform(-0.2,0.2)
    n1 = np.random.uniform(-0.2,0.2)
    drr0 = splev(s0,track.raceline_s,der=1)
    '''
    heading0 = np.arctan2(drr0[1],drr0[0])
    drr1 = splev(s1,track.raceline_s,der=1)
    heading1 = np.arctan2(drr0[1],drr0[0])
    '''

    x0 = np.array((s0,v0,n0,0,0))
    x1 = np.array((s1,v1,n1,0,0))
    delta_x = x0 - x1
    is_in_collision = np.abs(delta_x[0])<0.3*1.1 and np.abs(delta_x[2])<0.19*1.1
    is_outside = track.isOutsideCurv(x0) or track.isOutsideCurv(x1)
    if (is_in_collision or is_outside):
        return getRandomInitialStatePair(leader,track)
    else:
        cart0 = KinematicBicycleFrenetSimulator.curv2CartTrack(x0,track)
        cart1 = KinematicBicycleFrenetSimulator.curv2CartTrack(x1,track)
        curv0 = KinematicBicycleFrenetSimulator.cart2CurvTrack(cart0,track)
        curv1 = KinematicBicycleFrenetSimulator.cart2CurvTrack(cart1,track)
        if (curv0[0] == curv1[0]):
            breakpoint()
        return tuple(cart0[:4]),tuple(cart1[:4])

def configHelper(config):
    config_extensions = config.getElementsByTagName('extensions')[0]
    config_cars = config.getElementsByTagName('cars')[0]
    config_car0 = config_cars.getElementsByTagName('car')[0]
    config_car1 = config_cars.getElementsByTagName('car')[1]
    config_car0_controller = config_car0.getElementsByTagName('controller')[0]
    config_car1_controller = config_car1.getElementsByTagName('controller')[0]
    return config_car0, config_car0_controller, config_car1, config_car1_controller

def exploit_sine_dense():
    index = 0
    setup_vec = ['mpc','coordinative', 'adversarial','baseline']
    for i in range(50):
        for setup in setup_vec:
            s0,s1 = getRandomInitialStatePair(leader=0,track=track)
            for (car0_x0, car1_x0) in [(s0,s1),(s1,s0)]:
                config = deepcopy(original_config)
                config_car0, config_car0_controller, config_car1, config_car1_controller = configHelper(config)
                config_car0.getElementsByTagName('init_states')[0].childNodes[0].data = str(car0_x0)
                config_car1.getElementsByTagName('init_states')[0].childNodes[0].data = str(car1_x0)

                # controller name
                config_car0_controller.childNodes[1].childNodes[0].data = 'iLQGameCarController'
                config_car0_controller.attributes['blocking_control'] =  'True'
                config_car0_controller.attributes['alpha'] =  '1.0'

                if (setup == 'mpc'):
                    config_car1_controller.childNodes[1].childNodes[0].data = 'iLQGameSoloCarController'
                elif (setup == 'coordinative'):
                    config_car1_controller.childNodes[1].childNodes[0].data = 'iLQGameCarController'
                    config_car1_controller.attributes['Qop1'] =  '-4'
                    config_car1_controller.attributes['Qop2'] =  '-4'
                elif (setup == 'adversarial'):
                    config_car1_controller.childNodes[1].childNodes[0].data = 'iLQGameCarController'
                    config_car1_controller.attributes['Qop1'] =  '4'
                    config_car1_controller.attributes['Qop2'] =  '4'
                elif (setup == 'baseline'):
                    config_car1_controller.childNodes[1].childNodes[0].data = 'iLQGameCarController'
                    config_car1_controller.attributes['Qop1'] =  '0'
                    config_car1_controller.attributes['Qop2'] =  '0'
                else:
                    print('error')

                with open(config_folder+'exp%d.xml'%(index),'w') as f:
                    config.writexml(f)
                index += 1

    print('generated %d configs'%index)

def exploit_triangle_dense():
    return exploit_sine_dense()
def exploit_nascar_dense():
    return exploit_sine_dense()

def sine_mpc_dense():
    index = 0
    setup_vec = ['coordinative', 'adversarial','baseline']
    for i in range(50):
        for setup in setup_vec:
            s0,s1 = getRandomInitialStatePair(leader=0,track=track)
            for (car0_x0, car1_x0) in [(s0,s1),(s1,s0)]:
                config = deepcopy(original_config)
                config_car0, config_car0_controller, config_car1, config_car1_controller = configHelper(config)
                config_car0.getElementsByTagName('init_states')[0].childNodes[0].data = str(car0_x0)
                config_car1.getElementsByTagName('init_states')[0].childNodes[0].data = str(car1_x0)


                if (setup == 'coordinative'):
                    config_car0_controller.childNodes[1].childNodes[0].data = 'iLQGameCarController'
                    config_car0_controller.attributes['Qop1'] =  '-4'
                    config_car0_controller.attributes['Qop2'] =  '-4'
                elif (setup == 'adversarial'):
                    config_car0_controller.childNodes[1].childNodes[0].data = 'iLQGameCarController'
                    config_car0_controller.attributes['Qop1'] =  '4'
                    config_car0_controller.attributes['Qop2'] =  '4'
                elif (setup == 'baseline'):
                    config_car0_controller.childNodes[1].childNodes[0].data = 'iLQGameCarController'
                    config_car0_controller.attributes['Qop1'] =  '0'
                    config_car0_controller.attributes['Qop2'] =  '0'
                else:
                    print('error')

                # controller name
                config_car1_controller.childNodes[1].childNodes[0].data = 'iLQGameSoloCarController'

                with open(config_folder+'exp%d.xml'%(index),'w') as f:
                    config.writexml(f)
                index += 1

    print('generated %d configs'%index)

def triangle_mpc_dense():
    return sine_mpc_dense()
def nascar_mpc_dense():
    return sine_mpc_dense()

def four_algo():
    index = 0
    setup_vec = ['mppi-ibr_mppi','mppi_ilqr','mppi-ibr_ilqgame']
    track_name_vec = ['nascar_saved','triangle_saved','sine']

    for track_name in track_name_vec:
        config = deepcopy(original_config)
        config_track = config.getElementsByTagName('track')[0]
        config_track.childNodes[0].data = track_name
        track = TrackFactory.build(main=None,config=config_track)
        track.init()
        for i in range(50):
            for setup in setup_vec:
                s0,s1 = getRandomInitialStatePair(leader=0,track=track)
                for (car0_x0, car1_x0) in [(s0,s1),(s1,s0)]:
                    config = deepcopy(original_config)
                    config_car0, config_car0_controller, config_car1, config_car1_controller = configHelper(config)
                    config_car0.getElementsByTagName('init_states')[0].childNodes[0].data = str(car0_x0)
                    config_car1.getElementsByTagName('init_states')[0].childNodes[0].data = str(car1_x0)
                    config_track = config.getElementsByTagName('track')[0]
                    config_track.childNodes[0].data = track_name

                    if (setup == 'mppi-ibr_mppi'):
                        config_car0_controller.childNodes[1].childNodes[0].data = 'MppiFrenetCarController'
                        config_car0_controller.attributes['horizon'] =  '20'
                        config_car0_controller.attributes['samples_count'] =  '1024'
                        config_car0_controller.attributes['ibr_iter'] =  '3'

                        config_car1_controller.childNodes[1].childNodes[0].data = 'MppiFrenetCarController'
                        config_car1_controller.attributes['horizon'] =  '20'
                        config_car1_controller.attributes['samples_count'] =  '1024'

                    elif (setup == 'mppi_ilqr'):
                        config_car0_controller.childNodes[1].childNodes[0].data = 'MppiFrenetCarController'
                        config_car0_controller.attributes['horizon'] =  '20'
                        config_car0_controller.attributes['samples_count'] =  '1024'

                        config_car1_controller.childNodes[1].childNodes[0].data = 'iLQGameSoloCarController'

                    elif (setup == 'mppi-ibr_ilqgame'):
                        config_car0_controller.childNodes[1].childNodes[0].data = 'MppiFrenetCarController'
                        config_car0_controller.attributes['horizon'] =  '20'
                        config_car0_controller.attributes['samples_count'] =  '1024'
                        config_car0_controller.attributes['ibr_iter'] =  '3'

                        config_car1_controller.childNodes[1].childNodes[0].data = 'iLQGameCarController'

                    else:
                        print('error')

                    with open(config_folder+'exp%d.xml'%(index),'w') as f:
                        config.writexml(f)
                    index += 1

    print('generated %d configs'%index)
def four_algo_update_prediction():
    return four_algo()


if __name__=='__main__':
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

    eval(f'{name}()')
