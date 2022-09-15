import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from common import *
import seaborn as sns
import matplotlib.pyplot as plt
import pandas

if __name__ == '__main__':
    # read textlog
    if (len(sys.argv) == 2):
        name = sys.argv[1]
    else:
        name = 'grid16'
        print_info("defaulting to ",name)
        #print_error("you must specify a experiment name, which is the name of log/config folder")
    textlog_filename = os.path.join(os.path.dirname(sys.argv[0]),'../../log',name,'textlog.txt')
    with open(textlog_filename,'r') as f:
        lines = f.readlines()

    # parse log file
    labels = ['experiment_name' , 'config_file_name' , 'log_name' , 'laps' , 'enable_cvar','cvar_A' , 'cvar_a' , 'cvar_Cu' , 'laptime_mean' , 'laptime_stddev' , 'boundary_violation' , 'obstacle_violation','noise_type','noise_magnitude'] 
    data_dict = {}
    for label in labels:
        data_dict[label] = []

    sep_lines = []
    for line in lines:
        if line[0] == '#':
            continue
        sep_line = line.split(',')
        for label,text in zip(labels,sep_line):
            try:
                data_dict[label].append( eval(text) )
            except (NameError,SyntaxError):
                data_dict[label].append( text )

    cvar_enable_data = np.array(data_dict['enable_cvar'])
    cvar_a_data = np.array(data_dict['cvar_a'])
    cvar_Cu_data = np.array(data_dict['cvar_Cu'])
    boundary_data = np.array(data_dict['boundary_violation'])
    obstacle_data = np.array(data_dict['obstacle_violation'])
    noise_type_data = np.array(data_dict['noise_type'])
    noise_mag_data = np.array(data_dict['noise_magnitude'])

    text = '(boundary_data + obstacle_data)/101*11'
    value = eval(text)

    noise_vec = np.array([0.1,0.2,0.3,0.4,0.5])
    noise_type_vec = ['normal','uniform','impulse']

    cvar_normal = np.zeros(5)
    cvar_uniform = np.zeros(5)
    cvar_impulse = np.zeros(5)

    baseline_normal = np.zeros(5)
    baseline_uniform = np.zeros(5)
    baseline_impulse = np.zeros(5)

    for i in range(len(noise_vec)):
        for noise_type in noise_type_vec:
            for is_cvar in [True,False]:
                noise = noise_vec[i]
                mask1 = np.isclose(noise_mag_data,noise)
                mask2 = [val==noise_type for val in noise_type_data]
                mask3 = [val==is_cvar for val in cvar_enable_data]
                mask = np.logical_and(mask1,mask2)
                mask = np.logical_and(mask,mask3)
                assert sum(mask)==1
                if (is_cvar and noise_type == 'normal'):
                    try:
                        cvar_normal[i] = value[mask]
                        #idx = np.nonzero(mask)[0][0]
                        #print('cvar normal',noise,' ',value[mask],' id=',idx)
                    except ValueError:
                        cvar_normal[i] = -1
                elif (is_cvar and noise_type == 'uniform'):
                    try:
                        cvar_uniform[i] = value[mask]
                    except ValueError:
                        cvar_uniform[i] = -1
                elif (is_cvar and noise_type == 'impulse'):
                    try:
                        cvar_impulse[i] = value[mask]
                    except ValueError:
                        cvar_impulse[i] = -1
                elif ((not is_cvar) and noise_type == 'normal'):
                    try:
                        baseline_normal[i] = value[mask]
                        #idx = np.nonzero(mask)[0][0]
                        #print('baseline normal',noise,' ',value[mask],' id=',idx)
                    except ValueError:
                        baseline_normal[i] = -1
                elif ((not is_cvar) and noise_type == 'uniform'):
                    try:
                        baseline_uniform[i] = value[mask]
                    except ValueError:
                        baseline_uniform[i] = -1
                elif ((not is_cvar) and noise_type == 'impulse'):
                    try:
                        baseline_impulse[i] = value[mask]
                    except ValueError:
                        baseline_impulse[i] = -1


    breakpoint()
    noise_vec_text = ['%.1f'%val for val in noise_vec]
    mask = cvar_normal > 0
    plt.plot(noise_vec[mask], cvar_normal[mask], 'r--',label='cvar normal')
    mask = cvar_uniform > 0
    plt.plot(noise_vec[mask], cvar_uniform[mask], 'g--',label='cvar uniform')
    mask = cvar_impulse > 0
    plt.plot(noise_vec[mask], cvar_impulse[mask], 'b--',label='cvar impulse')
    mask = baseline_normal > 0
    plt.plot(noise_vec[mask], baseline_normal[mask], 'r-',label='baseline normal')
    mask = baseline_uniform > 0
    plt.plot(noise_vec[mask], baseline_uniform[mask], 'g-',label='baseline uniform')
    mask = baseline_impulse > 0
    plt.plot(noise_vec[mask], baseline_impulse[mask], 'b-',label='baseline impulse')
    plt.title(text)
    plt.xlabel('noise')
    plt.ylabel('collision')
    plt.legend()
    plt.show()

    

