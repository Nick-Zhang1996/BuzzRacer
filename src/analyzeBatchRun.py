import numpy as np
import matplotlib.pyplot as plt
from xml.dom import minidom
import xml.etree.ElementTree as ET
from track import TrackFactory

labels = "experiment_name , config_file_name , log_name , laps , Qop1, Qop2, start_lead_i_j, end_lead_i_j, laptime_mean_i , boundary_violation_i ,laptime_mean_j , boundary_violation_j , opponent_col"
labels = [val.strip() for val in labels.split(',')]

name = 'aggressive_baseline'
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
    data_dict['controller_name'] = []
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
        #Qop1_blocking = eval(config_controller.attributes['Qop1_blocking'].nodeValue)
        #data_dict['Qop1_blocking'].append(Qop1_blocking)
        controller_name = config_controller.childNodes[1].childNodes[0].data
        data_dict['controller_name'].append(controller_name)

print(f'failed runs: {failed_runs_count}, total runs: {total_runs}')

for key in data_dict.keys():
    if (not isinstance(data_dict[key][0],str)):
        data_dict[key] = np.array(data_dict[key])

# Analyze data

data_dict['start_lead_i_j'] = (data_dict['start_lead_i_j'] + track.raceline_len_m/2)%track.raceline_len_m - track.raceline_len_m/2
data_dict['end_lead_i_j'] = (data_dict['end_lead_i_j'] + track.raceline_len_m/2)%track.raceline_len_m - track.raceline_len_m/2


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

def printTableEntry(controller_name):
    mask = [val == controller_name for val in data_dict['controller_name']]

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

    print(f'{controller_name:<30}\t, {follow_win_ratio:.2f}\t\t,{lead_win_ratio:.2f}\t\t,{overall_win_ratio:.2f}\t\t,{boundary_col_i:.2f}\t,{boundary_col_j:.2f}\t,{opponent_col:.2f}\t, {np.sum(mask)}')

print(f'{"controller":<30} \t, follow win \t, lead win \t,overall win \t, i out\t, j out\t, col\t,total runs')
printTableEntry('iLQGameCarController')
printTableEntry('AggressiveCarController')
