import pickle
import numpy as np
import matplotlib.pyplot as plt

with open('orca_training_metric.p','rb') as f:
    data = pickle.load(f)
with open('rcp_training_metric.p','rb') as f:
    data = pickle.load(f)

data = np.array(data)
p1_col, p2_col, p1_over, p2_over = data.T
#episode = list(range(20,24060,20*(int(24060/100/20))))
episode = list(range(len(p1_col)))
plt.plot(episode, p1_col)
plt.plot(episode, p2_col)
plt.show()
