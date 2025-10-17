# run a single experiment from a batch 
from common import *
from run import Main
import glob
import sys

if __name__ == '__main__':
    # usage: python batchRunSingle.py batch_name id
    if (len(sys.argv) == 3):
        name = sys.argv[1]
        count = int(sys.argv[2])
    else:
        print_error('''you must specify a folder name under configs/, and an id, usage:
    # usage: python batchRunSingle.py batch_name id''')
    config_filename = './configs/'+name+f'/exp{count}.xml'
    experiment = Main(config_filename)
    experiment.experiment_name = name
    experiment.init()
    experiment.run()
