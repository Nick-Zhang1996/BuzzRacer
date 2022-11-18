import pickle
import numpy as np
import matplotlib.pyplot as plt

with open('orca_training_metric.p','rb') as f:
    data = pickle.load(f)
#with open('rcp_training_metric.p','rb') as f:
#    data = pickle.load(f)

data = np.array(data)
p1_col, p2_col, p1_over, p2_over = data.T
episode = list(range(20,29980,20*(int(29980/100/20))))
episode = list(range(len(p1_col)))
plt.plot( p1_col)
plt.plot( p2_col)
plt.show()
