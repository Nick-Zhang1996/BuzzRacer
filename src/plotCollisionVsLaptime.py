import sys
from common import *
import pickle
import matplotlib.pyplot as plt

# same injected
with open("../log/kinematics_results/debug_dict8.p", 'rb') as f:
    data_injected = pickle.load(f)
# same terminal
with open("../log/kinematics_results/debug_dict9.p", 'rb') as f:
    data_terminal = pickle.load(f)
# ccmppi
with open("../log/kinematics_results/debug_dict10.p", 'rb') as f:
    data_ccmppi = pickle.load(f)


