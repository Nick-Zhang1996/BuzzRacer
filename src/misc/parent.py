import subprocess
from common import *
command = "python3 child.py"
process = subprocess.Popen(command, shell=True, stdout=None)
process.wait()

while (process.returncode !=0):
    num_lines = sum(1 for line in open('testlog.txt'))
    print_ok("resuming from %d"%(num_lines))
    command = "python3 child.py --skip %d"%(num_lines)
    process = subprocess.Popen(command, shell=True, stdout=None)
    process.wait()




