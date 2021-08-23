import sys
from common import *
import pickle
import matplotlib.pyplot as plt

with open("../log/kinematics_results/debug_dict1.p", 'rb') as f:
    data = pickle.load(f)

print(data)

