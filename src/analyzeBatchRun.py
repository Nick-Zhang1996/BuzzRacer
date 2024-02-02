import numpy as np
import matplotlib.pyplot as plt
from xml.dom import minidom
import xml.etree.ElementTree as ET
from track import TrackFactory

labels = "experiment_name , config_file_name , log_name , laps , Qop1, Qop2, start_lead_i_j, end_lead_i_j, laptime_mean , laptime_stddev , boundary_violation , obstacle_violation"
labels = [val.strip() for val in labels.split(',')]

name = 'blocking_exploit'
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
    data_dict['Qop1_blocking'] = []
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
        config_controller = config_car0.getElementsByTagName('controller')[0]
        Qop1_blocking = eval(config_controller.attributes['Qop1_blocking'].nodeValue)
        data_dict['Qop1_blocking'].append(Qop1_blocking)

print(f'failed runs: {failed_runs_count}, total runs: {total_runs}')

for key in data_dict.keys():
    if (not isinstance(data_dict[key][0],str)):
        data_dict[key] = np.array(data_dict[key])

# Analyze data

data_dict['start_lead_i_j'] = (data_dict['start_lead_i_j'] + track.raceline_len_m/2)%track.raceline_len_m - track.raceline_len_m/2
data_dict['end_lead_i_j'] = (data_dict['end_lead_i_j'] + track.raceline_len_m/2)%track.raceline_len_m - track.raceline_len_m/2

# table 1: Qop -> start/end lead
print('Qop1 \t, Qop2 \t, start \t, end   \t, gain')

def printTableEntry(Qop1,Qop2):
    mask = np.logical_and( data_dict['Qop1']==Qop1,data_dict['Qop2']==Qop2 )
    mean_start_lead = np.mean(data_dict['start_lead_i_j'][mask])
    mean_end_lead = np.mean(data_dict['end_lead_i_j'][mask])
    mean_gain = mean_end_lead - mean_start_lead
    print(f'{Qop1} \t, {Qop2} \t, {mean_start_lead:.4f} \t, {mean_end_lead:.4f} \t, {mean_gain:.4f}')

# table 2: agent i/j
mean_start_lead = np.mean(data_dict['start_lead_i_j'])
mean_end_lead = np.mean(data_dict['end_lead_i_j'])
mean_gain = mean_end_lead - mean_start_lead
print(f'mean gain: {mean_gain}')
printTableEntry(0,0)
printTableEntry(4,0)
printTableEntry(-4,0)
printTableEntry(0,-4)
printTableEntry(0,4)
printTableEntry(-4,-4)
printTableEntry(4,4)
printTableEntry(4,-4)
printTableEntry(-4,4)

# table 3: end gain vs lead gain
'''
plt.scatter(data_dict['start_lead_i_j'], data_dict['end_lead_i_j'])
plt.show()
'''

# table 4: gain:
gain = data_dict['end_lead_i_j'] - data_dict['start_lead_i_j']
win_i = np.sum(data_dict['end_lead_i_j']>0)
win_j = np.sum(data_dict['end_lead_i_j']<0)
print(f'win_i: {win_i}, win_j: {win_j}')

# table 5: when starting from behind, Qop vs finish ratio
def printTable5Entry(Qop1,Qop2):
    mask = np.logical_and( data_dict['Qop1']==Qop1,data_dict['Qop2']==Qop2 )
    follow_mask = np.logical_and( mask , data_dict['start_lead_i_j']<0 )
    follow_win_mask = np.logical_and( follow_mask , data_dict['end_lead_i_j']>0 )
    follow_win_ratio = np.sum(follow_win_mask)/np.sum(follow_mask)

    lead_mask = np.logical_and( mask , data_dict['start_lead_i_j']>0 )
    lead_win_mask = np.logical_and( lead_mask , data_dict['end_lead_i_j']>0 )
    lead_win_ratio = np.sum(lead_win_mask)/np.sum(lead_mask)

    overall_win_mask = np.logical_and( mask , data_dict['end_lead_i_j']>0 )
    overall_win_ratio = np.sum(overall_win_mask)/np.sum(mask)
    print(f'{Qop1} \t, {Qop2} \t, {follow_win_ratio:.2f}\t\t,{lead_win_ratio:.2f}\t\t,{overall_win_ratio:.2f}\t\t,{np.sum(mask)}')

def printTable5Entry_alt(Qop1,Qop2):
    mask = np.logical_and( data_dict['Qop1']==Qop1,data_dict['Qop2']==Qop2 )
    follow_mask = np.logical_and( mask , data_dict['start_lead_i_j']>0 )
    follow_win_mask = np.logical_and( follow_mask , data_dict['end_lead_i_j']<0 )
    follow_win_ratio = np.sum(follow_win_mask)/np.sum(follow_mask)

    lead_mask = np.logical_and( mask , data_dict['start_lead_i_j']<0 )
    lead_win_mask = np.logical_and( lead_mask , data_dict['end_lead_i_j']<0 )
    lead_win_ratio = np.sum(lead_win_mask)/np.sum(lead_mask)

    overall_win_mask = np.logical_and( mask , data_dict['end_lead_i_j']<0 )
    overall_win_ratio = np.sum(overall_win_mask)/np.sum(mask)
    print(f'{Qop1} \t, {Qop2} \t, {follow_win_ratio:.2f}\t\t,{lead_win_ratio:.2f}\t\t,{overall_win_ratio:.2f}\t\t,{np.sum(mask)}')

print(f'Qop1 \t, Qop2 \t, follow win \t, lead win \t,overall win \t, total runs')
print('Results for i')
printTable5Entry(0,0)
printTable5Entry(4,0)
printTable5Entry(-4,0)
printTable5Entry(0,-4)
printTable5Entry(0,4)
printTable5Entry(-4,-4)
printTable5Entry(4,4)
printTable5Entry(4,-4)
printTable5Entry(-4,4)

zero_lead = np.abs(data_dict['start_lead_i_j'])<1e-5
print(f'{np.sum(zero_lead)} zero leads')

# Table 6: analyze blocking controller performance
def printTable6Entry(qop):
    mask = data_dict['Qop1_blocking'] == qop
    follow_mask = np.logical_and( mask , data_dict['start_lead_i_j']<0 )
    follow_win_mask = np.logical_and( follow_mask , data_dict['end_lead_i_j']>0 )
    follow_win_ratio = np.sum(follow_win_mask)/np.sum(follow_mask)

    lead_mask = np.logical_and( mask , data_dict['start_lead_i_j']>0 )
    lead_win_mask = np.logical_and( lead_mask , data_dict['end_lead_i_j']>0 )
    lead_win_ratio = np.sum(lead_win_mask)/np.sum(lead_mask)

    overall_win_mask = np.logical_and( mask , data_dict['end_lead_i_j']>0 )
    overall_win_ratio = np.sum(overall_win_mask)/np.sum(mask)
    print(f'{qop} \t, {follow_win_ratio:.2f}\t\t,{lead_win_ratio:.2f}\t\t,{overall_win_ratio:.2f}\t\t,{np.sum(mask)}')

print('Qop=3')
printTable6Entry(3)
print('Qop=-3')
printTable6Entry(-3)
print('No exploiting')
printTable6Entry(-100)

# Table 7: look at experiments where exploiting controller has an advantage

'''
breakpoint()
mask_exploiting = data_dict['Qop1_blocking'] == -3
mask_win = data_dict['end_lead_i_j']>0
mask = np.logical_and(mask_exploiting, mask_win)
#qop_vec = [3,-3,-100], so if i -> -3, the next log is same initial state with -100
for index in np.nonzero(mask):
    # find config with identical initial states
    # without exploiting
    # that failed
    assert(data_dict['Qop1_blocking'][index+1] == -100)
    if (not mask_win[index+1]

'''
