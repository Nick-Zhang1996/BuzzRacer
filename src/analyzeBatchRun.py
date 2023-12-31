import numpy as np
import matplotlib.pyplot as plt

labels = "experiment_name , config_file_name , log_name , laps , Qop1, Qop2, start_lead_i_j, end_lead_i_j, laptime_mean , laptime_stddev , boundary_violation , obstacle_violation"
labels = [val.strip() for val in labels.split(',')]

with open('../log/batch_ilqgame/textlog.txt','r') as f:
    text = f.readlines()
    text = text[1:]
    data_dict = {}
    for label in labels:
        data_dict[label] = []
    for line in text:
        entry = line.split(',')
        for raw,label in zip(entry,labels):
            try:
                datum = eval(raw)
            except (NameError,SyntaxError):
                datum = raw
            data_dict[label].append(datum)

for key in data_dict.keys():
    if (not isinstance(data_dict[key][0],str)):
        data_dict[key] = np.array(data_dict[key])

# table 1: Qop -> start/end lead
print('Qop1 \t, Qop2 \t, start \t, end   \t, gain')

def printTableEntry(Qop1,Qop2):
    mask = np.logical_and( data_dict['Qop1']==Qop1,data_dict['Qop2']==Qop2 )
    mean_start_lead = np.mean(data_dict['start_lead_i_j'][mask])
    mean_end_lead = np.mean(data_dict['end_lead_i_j'][mask])
    mean_gain = mean_end_lead - mean_start_lead
    print(f'{Qop1} \t, {Qop2} \t, {mean_start_lead:.4f} \t, {mean_end_lead:.4f} \t, {mean_gain:.4f}')
printTableEntry(0,1)
printTableEntry(1,0)
printTableEntry(0,0)
printTableEntry(1,1)

# table 2: agent i/j
mean_start_lead = np.mean(data_dict['start_lead_i_j'])
mean_end_lead = np.mean(data_dict['end_lead_i_j'])
mean_gain = mean_end_lead - mean_start_lead
print(f'mean gain: {mean_gain}')

# table 3: end gain vs lead gain
'''
plt.scatter(data_dict['start_lead_i_j'], data_dict['end_lead_i_j'])
plt.show()
'''

# table 4: gain:
gain = data_dict['end_lead_i_j'] - data_dict['start_lead_i_j']
win_i = np.sum(gain>0)
win_j = np.sum(-gain>0)
print(f'win_i: {win_i}, win_j: {win_j}')

