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

if (len(sys.argv) == 2):
    name = sys.argv[1]
else:
    print_error("you must specify a folder name under configs/")

config_folder = './configs/' + name + '/'
config_filename = config_folder + 'master.xml'
original_config = minidom.parse(config_filename)

index = 0
start_pos = [(0.2,0.1, radians(0), 3),
(0.4,0.1, radians(0), 3)]
Qop = [0,2,4]


for s0 in start_pos:
    for s1 in start_pos:
        if (s0 == s1):
            continue
        for q0 in Qop:
            for q1 in Qop:
                config = deepcopy(original_config)
                config_extensions = config.getElementsByTagName('extensions')[0]
                config_cars = config.getElementsByTagName('cars')[0]
                config_car0 = config_cars.getElementsByTagName('car')[0]
                config_car1 = config_cars.getElementsByTagName('car')[1]
                config_controller = config_car1.getElementsByTagName('controller')[0]
                #attrs = config_controller.attributes.items()
                config_controller.attributes['Qop1'] =  str(q0)
                config_controller.attributes['Qop2'] =  str(q1)

                config_car0.getElementsByTagName('init_states')[0].childNodes[0].data = str(s0)
                config_car1.getElementsByTagName('init_states')[0].childNodes[0].data = str(s1)

                with open(config_folder+'exp%d.xml'%(index),'w') as f:
                    config.writexml(f)
                index += 1

print('generated %d configs'%index)
