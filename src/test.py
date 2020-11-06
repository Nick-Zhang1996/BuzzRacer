import numpy as np
from time import time

'''
A = np.random.rand(5,5)
print(A)
tic = time()
B = np.linalg.matrix_power(A,50)
tac = time()
print(B,1.0/(tac-tic))
'''

import cvxopt
from cvxopt import matrix, solvers
Q = 2*matrix([ [2, .5], [.5, 1] ])
p = matrix([1.0, 1.0])
G = matrix([[-1.0,0.0],[0.0,-1.0]])
h = matrix([0.0,0.0])
A = matrix([1.0, 1.0], (1,2))
b = matrix(1.0)
cvxopt.solvers.options['show_progress'] = False

times = []
for i in range(100):
    tic = time()
    sol=solvers.qp(Q, p, G, h, A, b)
    times.append(time()-tic)

print(sol['x'],1.0/(np.mean(times)))
