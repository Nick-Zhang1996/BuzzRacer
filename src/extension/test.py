import os
import sys
print(__file__) # relative filename
print(os.path.dirname(__file__)) # directory name that contains the file
print(os.path.join(os.path.dirname(__file__),'..')) # parent directory for this file
print(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # abs path of this file's parent

