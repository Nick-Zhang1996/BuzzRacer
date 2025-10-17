# nanny to restart benchmark.py when it fails
import subprocess
from common import *
from time import time
t0 = time()
skip = 0
command = "python benchmark.py --skip %d"%(skip)
print_warning("skipping %d"%(skip))
process = subprocess.Popen(command, shell=True, stdout=None)
process.wait()

while (process.returncode !=0):
    num_lines = sum(1 for line in open('log.txt'))
    print_ok("[Nanny] -------  ---------------- -------")
    print_ok("[Nanny] -------  resuming from %d -------"%(num_lines))
    print_ok("[Nanny] -------  ---------------- -------")
    with open("nanny.log",'w') as f:
        f.write("[%.2f] resuming from %d"%(time(),num_lines))
    command = "python benchmark.py --skip %d"%(num_lines)
    process = subprocess.Popen(command, shell=True, stdout=None)
    process.wait()




