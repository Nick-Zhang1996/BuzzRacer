import matplotlib.pyplot as plt
import numpy as np
import pickle

filename = "test2/exp_state.p"
infile = open(filename,'rb')
data = pickle.load(infile)
data = np.array(data)
plt.plot(data[30:,3])
plt.show()
