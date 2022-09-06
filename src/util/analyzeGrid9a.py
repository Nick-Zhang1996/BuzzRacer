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
        name = 'grid9'
        print_info("defaulting to ",name)
        #print_error("you must specify a experiment name, which is the name of log/config folder")
    textlog_filename = os.path.join(os.path.dirname(sys.argv[0]),'../../log',name,'textlog9a.txt')
    with open(textlog_filename,'r') as f:
        lines = f.readlines()

    # parse log file
    labels = ['experiment_name' , 'config_file_name' , 'log_name' , 'laps' , 'enable_cvar','cvar_A' , 'cvar_a' , 'cvar_Cu' , 'laptime_mean' , 'laptime_stddev' , 'boundary_violation' , 'obstacle_violation'] 
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
    # assemble matrix, x:cvar_a, y:cvar_Cu, val:total_collision

    cvar_enable = np.array(data_dict['enable_cvar'])
    cvar_a_data = np.array(data_dict['cvar_a'])
    cvar_A_data = np.array(data_dict['cvar_A'])
    cvar_Cu_data = np.array(data_dict['cvar_Cu'])
    boundary_data = np.array(data_dict['boundary_violation'])
    obstacle_data = np.array(data_dict['obstacle_violation'])

    text = '(boundary_data + obstacle_data)/101*11'
    value = eval(text)

    enable_cvar = True
    cvar_a_vec = np.linspace(0.1,0.9,5)
    cvar_A_vec = [4,7,10]

    grid_cvar = np.zeros((len(cvar_a_vec), len(cvar_A_vec)))
    #grid_baseline = np.zeros((len(cvar_a_vec), len(cvar_A_vec)))
    for i in range(len(cvar_a_vec)):
        for j in range(len(cvar_A_vec)):
            cvar_a = cvar_a_vec[i]
            cvar_A = cvar_A_vec[j]
            mask1 = np.isclose(cvar_a_data,cvar_a)
            mask2 = np.isclose(cvar_A_data,cvar_A)
            mask = np.logical_and(mask1,mask2)
            try:
                grid_cvar[i,j] = value[np.logical_and(mask,cvar_enable)]
            except ValueError:
                grid_cvar[i,j] = -1
            #grid_baseline[i,j] = value[np.logical_and(mask,np.logical_not(cvar_enable))]

    #mean_baseline = np.mean(grid_baseline)
    #std_baseline = np.std(grid_baseline)
    # for noise=0.2
    mean_baseline = 367.095
    std_baseline = 9.081
    print('baseline mean = %.3f, std = %.3f'%(mean_baseline, std_baseline))
    advantage = (grid_cvar/mean_baseline)
    advantage = pandas.DataFrame(data=advantage, index=cvar_a_vec, columns=cvar_A_vec)

    ax = sns.heatmap(advantage, annot=True,linewidth=0.5)
    plt.title(text)
    plt.xlabel('cvar_A')
    plt.ylabel('cvar_a')
    plt.show()

    

