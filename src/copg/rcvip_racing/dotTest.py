import numpy as np
a = [[1,2],[3,4]]
b = [[2,4],[5,6]]
# expected: 10,39
a = np.array(a)
b = np.array(b)
print('a',a.shape)
print('b',b.shape)
print(np.tensordot(a,b,1))
print(np.tensordot(a,b,1).shape)
