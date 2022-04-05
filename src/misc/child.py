import random
from time import sleep
from common import *
import argparse

# start with no. skip experiment, index start with 0
# e.g. skip = 3 means skip 0,1,2
parser = argparse.ArgumentParser()
parser.add_argument("--skip", help="number of experiments to skip", type=int, default=0)
args = parser.parse_args()
print_info("skipping %d experiments"%(args.skip))


random.seed()

for i in range(20):
    if (i<args.skip):
        continue

    sleep(0.1)
    if (random.random() > 0.9):
        print_error("random failure")
    else:
        print_ok("success %d"%(i))
        with open("testlog.txt",'a') as f:
            f.write("success %d\n"%(i))
exit(0)

