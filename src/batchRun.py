# run batch experiment
from common import *
from run import Main
import glob
import sys

skip = 0
if __name__ == '__main__':
    if (len(sys.argv) == 2):
        name = sys.argv[1]
    else:
        print_error("you must specify a folder name under configs/")

    pattern = './configs/'+name+'/exp*.xml'
    config_filename_vec = glob.glob(pattern)

    '''
    pattern1 = './configs/'+'grid8'+'/exp*.xml'
    config_filename_vec1 = glob.glob(pattern1)
    pattern2 = './configs/'+'grid11'+'/exp*.xml'
    config_filename_vec2 = glob.glob(pattern1)
    config_filename_vec = config_filename_vec1 + config_filename_vec2
    '''

    if (len(config_filename_vec) == 0):
        print_error("no file fitting pattern ",pattern)
    print_info('total configs: %d'%len(config_filename_vec))

    # TODO add progress report, suppress output
    print_info('skipping first %d exp'%(skip))
    count = 0
    for config_filename in config_filename_vec:
        if (count < skip):
            print_info('skipping ',config_filename)
            count += 1
            continue
        experiment = Main(config_filename)
        experiment.experiment_name = name
        experiment.init()
        experiment.run()
