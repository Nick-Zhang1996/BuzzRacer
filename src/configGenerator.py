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
#cvar_a_vec = np.linspace(0.1,0.9,3)
#cvar_Cu_vec = [0,0.5,1,2,5]
#cvar_a_vec = np.linspace(0.1,0.9,9)
#cvar_Cu_vec = np.linspace(0.1,0.9,9)

# grid 5,6, grid 6 use 0.1 noise, grid5 use 0.2 noise
#cvar_a_vec = [0.99,0.95,0.93]
#cvar_Cu_vec = np.linspace(0.5,0.9,5)

# grid 7, also has hand written tests with different noise level
#cvar_a_vec = [0.95]
#cvar_Cu_vec = [0.5]
#enable_cvar = True
#cvar_A_vec = [2,4,6,8,10]

# grid 9a
#enable_cvar = True
#cvar_a_vec = np.linspace(0.1,0.9,5)
#cvar_A_vec = [4,7,10]
#cvar_Cu = 0.5

# grid9b
#noise_vec = [0.3,0.5,0.7]

# grid 13
#cvar_a_vec = np.linspace(0.1,0.9,5)
#cvar_Cu_vec = [0.5,0.8,1.0,10,100,200]
#cvar_A = 10.0
#enable_cvar = True

# grid 15, 17
#cvar_a_vec = np.linspace(0.1,0.9,5)
#cvar_Cu_vec = np.linspace(0.6,1.0,5)
#cvar_A = 10.0
#enable_cvar = True

# grid 16
# baseline vs cvar
# search noise level
# search noise type
cvar_a = 0.5
cvar_Cu = 0.5
cvar_A = 10.0
enable_cvar_vec = [True,False]
noise_vec = [0.1,0.2,0.3,0.4,0.5]
noise_type_vec = ['normal','uniform','impulse']

for enable_cvar in enable_cvar_vec:
    for noise_type in noise_type_vec:
        for noise in noise_vec:
            config = deepcopy(original_config)
            config_extensions = config.getElementsByTagName('extensions')[0]
            for config_extension in config_extensions.getElementsByTagName('extension'):
                if config_extension.getAttribute('handle') == 'simulator':
                    config_extension.attributes['state_noise_magnitude'] = str([noise]*6)
                    config_extension.attributes['state_noise_type'] =  str(noise_type)
            config_cars = config.getElementsByTagName('cars')[0]
            config_car = config_cars.getElementsByTagName('car')[0]
            config_controller = config_car.getElementsByTagName('controller')[0]
            attrs = config_controller.attributes.items()

            config_controller.attributes['enable_cvar'] =  str(enable_cvar)
            config_controller.attributes['cvar_Cu'] =  str(cvar_Cu)
            config_controller.attributes['cvar_a'] =   str(cvar_a)
            config_controller.attributes['cvar_A'] =   str(cvar_A)
            config_controller.attributes['state_noise_magnitude'] =  str([noise]*6) 
            config_controller.attributes['state_noise_type'] =  str(noise_type)

            with open(config_folder+'exp%d.xml'%(index),'w') as f:
                config.writexml(f)
            index += 1

print('generated %d configs'%index)
