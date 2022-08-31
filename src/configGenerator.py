# Example: load a config, modify as needed, then save xml
# to run batch experiments, you should write a file like this to generate all configs needed, then call 
# python batchExperiment.py folder_of_config
from common import *
from xml.dom import minidom
import xml.etree.ElementTree as ET
import numpy as np
from copy import deepcopy
import sys

if (len(sys.argv) == 2):
    name = sys.argv[1]
else:
    print_error("you must specify a folder name under configs/")

config_folder = './configs/' + name + '/'
config_filename = config_folder + 'master.xml'
original_config = minidom.parse(config_filename)

index = 0
cvar_a_vec = np.linspace(0.1,0.9,9)
cvar_Cu_vec = [0,1,5,10,20,30]

for cvar_a in cvar_a_vec:
    for cvar_Cu in cvar_Cu_vec:
        config = deepcopy(original_config)
        # get cvar_a
        config_cars = config.getElementsByTagName('cars')[0]
        config_car = config_cars.getElementsByTagName('car')[0]
        config_controller = config_car.getElementsByTagName('controller')[0]
        attrs = config_controller.attributes.items()

        config_controller.attributes['cvar_Cu'] =  str(cvar_Cu)
        config_controller.attributes['cvar_a'] =   str(cvar_a)
        with open(config_folder+'exp%d.xml'%(index),'w') as f:
            config.writexml(f)
        index += 1
