from xml.dom import minidom
import xml.etree.ElementTree as ET

log = '../log/batch_ilqgame_solo/textlog.txt'
newlog = '../log/batch_ilqgame_solo/new_textlog.txt'
print('opening log ' + log)
failed_runs_count = 0
new_textlog = []
with open(log, 'r') as f:
    text = f.readlines()
    # text = text[1:]
    for line in text:
        entry = line.split(',')
        # open config in position 1
        config = minidom.parse(entry[1])

        # read Qop1, Qop2
        config_extensions = config.getElementsByTagName('extensions')[0]
        config_cars = config.getElementsByTagName('cars')[0]
        config_car0 = config_cars.getElementsByTagName('car')[0]
        config_car1 = config_cars.getElementsByTagName('car')[1]
        config_controller = config_car0.getElementsByTagName('controller')[0]
        Qop1 = config_controller.getAttribute('Qop1')
        Qop2 = config_controller.getAttribute('Qop2')
        entry[4] = Qop1
        entry[5] = Qop2
        # correct lead_i_j sign
        entry[6] = str(-eval(entry[6]))
        entry[7] = str(-eval(entry[7]))
        new_textlog.append(','.join(entry))

with open(newlog, 'w') as f:
    f.writelines(new_textlog)
