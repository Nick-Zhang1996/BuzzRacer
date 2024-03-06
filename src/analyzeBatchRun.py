import sys
import numpy as np
import matplotlib.pyplot as plt
from xml.dom import minidom
import xml.etree.ElementTree as ET
from track import TrackFactory

if (len(sys.argv) == 2):
    name = sys.argv[1]
else:
    print_error("you must specify a folder name under configs/")

labels = "experiment_name , config_file_name , log_name , laps , Qop1_obs, Qop2_obs, start_lead_i_j, end_lead_i_j, laptime_mean_i , boundary_violation_i ,laptime_mean_j , boundary_violation_j , opponent_col"
labels = [val.strip() for val in labels.split(',')]

config_folder = './configs/' + name + '/'
config_filename = config_folder + 'master.xml'
original_config = minidom.parse(config_filename)

config_track= original_config.getElementsByTagName('track')[0]
track = TrackFactory.build(main=None,config=config_track)
track.init()

# Load textlog.txt and config xml 
log = '../log/'+name+'/textlog.txt'
print('opening log ' + log)
failed_runs_count = 0
with open(log,'r') as f:
    text = f.readlines()
    text = text[1:]
    data_dict = {}
    total_runs = len(text)
    for label in labels:
        data_dict[label] = []
    #data_dict['Qop1_blocking'] = []
    data_dict['car0_controller_name'] = []
    data_dict['car1_controller_name'] = []
    data_dict['blocking_control'] = []
    # alpha, aggressiveness
    data_dict['a0'] = []
    data_dict['a1'] = []
    # Qop1, Qop2 of car0
    data_dict['car0_Qop1'] = []
    data_dict['car0_Qop2'] = []
    data_dict['car1_Qop1'] = []
    data_dict['car1_Qop2'] = []
    for line in text:
        entry = line.split(',')
        if (eval(entry[-1]) == -1):
            failed_runs_count += 1
            continue
        for raw,label in zip(entry,labels):
            try:
                datum = eval(raw)
            except (NameError,SyntaxError):
                datum = raw
            data_dict[label].append(datum)
        config_filename = entry[1]
        config = minidom.parse(config_filename)
        config_cars = config.getElementsByTagName('cars')[0]
        config_car0 = config_cars.getElementsByTagName('car')[0]
        config_car1 = config_cars.getElementsByTagName('car')[1]
        config_car0_controller = config_car0.getElementsByTagName('controller')[0]
        config_car1_controller = config_car1.getElementsByTagName('controller')[0]
        #Qop1_blocking = eval(config_controller.attributes['Qop1_blocking'].nodeValue)
        #data_dict['Qop1_blocking'].append(Qop1_blocking)
        car0_controller_name = config_car0_controller.childNodes[1].childNodes[0].data
        car1_controller_name = config_car1_controller.childNodes[1].childNodes[0].data
        data_dict['car0_controller_name'].append(car0_controller_name)
        data_dict['car1_controller_name'].append(car1_controller_name)

        blocking = eval(config_car0_controller.attributes['blocking_control'].nodeValue)
        data_dict['blocking_control'].append(blocking)

        try:
            car0_Qop1 = eval(config_car0_controller.attributes['Qop1'].nodeValue)
            car0_Qop2 = eval(config_car0_controller.attributes['Qop2'].nodeValue)
        except KeyError:
            car0_Qop1 = 0
            car0_Qop2 = 0
        data_dict['car0_Qop1'].append(car0_Qop1)
        data_dict['car0_Qop2'].append(car0_Qop2)

        try:
            car1_Qop1 = eval(config_car1_controller.attributes['Qop1'].nodeValue)
            car1_Qop2 = eval(config_car1_controller.attributes['Qop2'].nodeValue)
        except KeyError:
            car1_Qop1 = 0
            car1_Qop2 = 0
        data_dict['car1_Qop1'].append(car1_Qop1)
        data_dict['car1_Qop2'].append(car1_Qop2)

        try:
            a0 = eval(config_car0_controller.attributes['alpha'].nodeValue)
            a1 = eval(config_car1_controller.attributes['alpha'].nodeValue)
        except KeyError:
            a0 = None
            a1 = None
        data_dict['a0'].append(a0)
        data_dict['a1'].append(a1)


print(f'failed runs: {failed_runs_count}, total runs: {total_runs}')

for key in data_dict.keys():
    if (not isinstance(data_dict[key][0],str)):
        data_dict[key] = np.array(data_dict[key])

# normalize data
data_dict['start_lead_i_j'] = (data_dict['start_lead_i_j'] + track.raceline_len_m/2)%track.raceline_len_m - track.raceline_len_m/2
data_dict['end_lead_i_j'] = (data_dict['end_lead_i_j'] + track.raceline_len_m/2)%track.raceline_len_m - track.raceline_len_m/2
data_dict['advance'] = data_dict['end_lead_i_j']-data_dict['start_lead_i_j']


# table 3: end gain vs lead gain
'''
plt.scatter(data_dict['start_lead_i_j'], data_dict['end_lead_i_j'])
plt.show()
'''

# table 4: win count:
gain = data_dict['end_lead_i_j'] - data_dict['start_lead_i_j']
win_i = np.sum(data_dict['end_lead_i_j']>0)
win_j = np.sum(data_dict['end_lead_i_j']<0)
print(f'win_i: {win_i}, win_j: {win_j}')

def printStats(text,mask):
    follow_mask = np.logical_and( mask , data_dict['start_lead_i_j']<0 )
    follow_win_mask = np.logical_and( follow_mask , data_dict['end_lead_i_j']>0 )
    follow_win_ratio = np.sum(follow_win_mask)/np.sum(follow_mask)

    lead_mask = np.logical_and( mask , data_dict['start_lead_i_j']>0 )
    lead_win_mask = np.logical_and( lead_mask , data_dict['end_lead_i_j']>0 )
    lead_win_ratio = np.sum(lead_win_mask)/np.sum(lead_mask)

    overall_win_mask = np.logical_and( mask , data_dict['end_lead_i_j']>0 )
    overall_win_ratio = np.sum(overall_win_mask)/np.sum(mask)

    boundary_col_i = np.mean(data_dict['boundary_violation_i'][mask])
    boundary_col_j = np.mean(data_dict['boundary_violation_j'][mask])

    opponent_col = np.mean(data_dict['opponent_col'][mask])
    advance = np.mean(data_dict['advance'][mask])

    print(text+f'\t{follow_win_ratio:.3f}\t,{lead_win_ratio:.3f}\t,{overall_win_ratio:.3f}\t,{boundary_col_i:.3f}\t,{boundary_col_j:.3f}\t,{opponent_col:.3f}\t,{advance:.3f}\t, {np.sum(mask)}')


def exploit_exploit():
    # a0, a1, lead/follow
    print(f'a0,a1 \t, follow win \t, lead win \t,overall win \t, i out\t, j out\t, col\t,total runs')
    def printTableEntry(a0,a1):
        #mask = [val == controller_name for val in data_dict['controller_name']]
        #mask = [val == blocking_control for val in data_dict['blocking_control']]
        mask_a0 = [val == a0 for val in data_dict['a0']]
        mask_a1 = [val == a1 for val in data_dict['a1']]
        mask = np.logical_and(mask_a0,mask_a1)
        printStats(f'a0={a0}, a1={a1}',mask)
    printTableEntry(0,0)
    printTableEntry(0.5,0)
    printTableEntry(1.0,0.5)
    printTableEntry(1.0,1.0)
    printTableEntry(1.0,0)

def ilqgame_mpc():
    #Qop1, Qop2
    print(f'qop0,qop1 \t, follow win \t, lead win \t,overall win \t, i out\t, j out\t, col\t,total runs')
    def printTableEntry(a0,a1):
        mask_a0 = [val == a0 for val in data_dict['car0_Qop1']]
        mask_a1 = [val == a1 for val in data_dict['car0_Qop2']]
        mask = np.logical_and(mask_a0,mask_a1)
        printStats(f'q0={a0}, q1={a1}',mask)
    printTableEntry(0,0)
    printTableEntry(-4,0)
    printTableEntry(4,0)
    printTableEntry(-4,-4)
    printTableEntry(4,4)

def ilqgame_single():
    #Qop1, Qop2
    print(f'qop0,qop1 \t, follow win \t, lead win \t,overall win \t, i out\t, j out\t, col\t,total runs')
    def printTableEntry(a0,a1):
        mask_a0 = [val == a0 for val in data_dict['car0_Qop1']]
        mask_a1 = [val == a1 for val in data_dict['car0_Qop2']]
        mask = np.logical_and(mask_a0,mask_a1)
        printStats(f'q0={a0}, q1={a1}',mask)
    printTableEntry(0,0)
    printTableEntry(-4,0)
    printTableEntry(4,0)
    printTableEntry(-4,-4)
    printTableEntry(4,4)

def exploit_triangle():
    # a0, a1, lead/follow
    print(f'a0,a1 \t, follow win \t, lead win \t,overall win \t, i out\t, j out\t, col\t,total runs')
    def printTableEntry(a0,a1):
        #mask = [val == controller_name for val in data_dict['controller_name']]
        #mask = [val == blocking_control for val in data_dict['blocking_control']]
        mask_a0 = [val == a0 for val in data_dict['a0']]
        mask_a1 = [val == a1 for val in data_dict['a1']]
        mask = np.logical_and(mask_a0,mask_a1)
        printStats(f'a0={a0}, a1={a1}',mask)
    printTableEntry(0,0)
    printTableEntry(0.5,0)
    printTableEntry(1.0,0.5)
    printTableEntry(1.0,1.0)
    printTableEntry(1.0,0)

def exploit_sine():
    # a0, a1, lead/follow
    print(f'a0,a1 \t, follow win \t, lead win \t,overall win \t, i out\t, j out\t, col\t,total runs')
    def printTableEntry(a0,a1):
        #mask = [val == controller_name for val in data_dict['controller_name']]
        #mask = [val == blocking_control for val in data_dict['blocking_control']]
        mask_a0 = [val == a0 for val in data_dict['a0']]
        mask_a1 = [val == a1 for val in data_dict['a1']]
        mask = np.logical_and(mask_a0,mask_a1)
        printStats(f'a0={a0}, a1={a1}',mask)
    printTableEntry(0,0)
    printTableEntry(0.5,0)
    printTableEntry(1.0,0.5)
    printTableEntry(1.0,1.0)
    printTableEntry(1.0,0)

def triangle_mpc():
    #Qop1, Qop2
    print(f'qop0,qop1 \t, follow win \t, lead win \t,overall win \t, i out\t, j out\t, col\t,total runs')
    def printTableEntry(a0,a1):
        mask_a0 = [val == a0 for val in data_dict['car0_Qop1']]
        mask_a1 = [val == a1 for val in data_dict['car0_Qop2']]
        mask = np.logical_and(mask_a0,mask_a1)
        printStats(f'q0={a0}, q1={a1}',mask)
    printTableEntry(0,0)
    printTableEntry(-4,0)
    printTableEntry(4,0)
    printTableEntry(-4,-4)
    printTableEntry(4,4)

def sine_mpc():
    #Qop1, Qop2
    print(f'qop0,qop1 \t, follow win \t, lead win \t,overall win \t, i out\t, j out\t, col\t,total runs')
    def printTableEntry(a0,a1):
        mask_a0 = [val == a0 for val in data_dict['car0_Qop1']]
        mask_a1 = [val == a1 for val in data_dict['car0_Qop2']]
        mask = np.logical_and(mask_a0,mask_a1)
        printStats(f'q0={a0}, q1={a1}',mask)
    printTableEntry(0,0)
    printTableEntry(-4,0)
    printTableEntry(4,0)
    printTableEntry(-4,-4)
    printTableEntry(4,4)

def sine_mpc_dense():
    #Qop1, Qop2
    print(f'qop0,qop1 \t,flow \t,lead \t,all \t,i out\t, j out\t, col\t,adv\t,total runs')
    def printTableEntry(a0,a1):
        mask_a0 = [val == a0 for val in data_dict['Qop1']]
        mask_a1 = [val == a1 for val in data_dict['Qop2']]
        mask = np.logical_and(mask_a0,mask_a1)
        printStats(f'q0={a0}, q1={a1}',mask)
    printTableEntry(0,0)
    printTableEntry(-4,-4)
    printTableEntry(4,4)

def triangle_mpc_dense():
    return sine_mpc_dense()
def nascar_mpc_dense():
    return sine_mpc_dense()

def exploit_sine_dense():
    print(f'qop0,qop1 \t,flow \t,lead \t,all\t,i out\t, j out\t, col\t,adv\t,total runs')
    def printTableEntry(a0,a1,use_mpc):
        mask_a0 = [val == a0 for val in data_dict['car1_Qop1']]
        mask_a1 = [val == a1 for val in data_dict['car1_Qop2']]
        mask = np.logical_and(mask_a0,mask_a1)
        if (use_mpc):
            mask_mpc = [val == 'iLQGameSoloCarController' for val in data_dict['car1_controller_name']]
        else:
            mask_mpc = [val != 'iLQGameSoloCarController' for val in data_dict['car1_controller_name']]
        mask = np.logical_and(mask,mask_mpc)
        printStats(f'{a0}/{a1} {"mpc" if use_mpc else "GT"}\t',mask)
    printTableEntry(0,0,True)
    printTableEntry(0,0,False)
    printTableEntry(-4,-4,False)
    printTableEntry(4,4, False)

def exploit_triangle_dense():
    return exploit_sine_dense()
def exploit_nascar_dense():
    return exploit_sine_dense()
    exploit_sine_dense()

if __name__=='__main__':
    eval(name+'()')
