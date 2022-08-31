# run batch experiment
from common import *
from run import Main
import glob
import sys

if __name__ == '__main__':
    if (len(sys.argv) == 2):
        name = sys.argv[1]
    else:
        print_error("you must specify a folder name under configs/")

    pattern = './configs/'+name+'/exp*.xml'
    config_filename_vec = glob.glob(pattern)
    if (len(config_filename_vec) == 0):
        print_error("no file fitting pattern ",pattern)
    print_info('total configs: %d',len(config_filename_vec))

    # TODO add progress report, suppress output
    for config_filename in config_filename_vec:
        experiment = Main(config_filename)
        experiment.experiment_name = name
        experiment.init()
        experiment.run()
