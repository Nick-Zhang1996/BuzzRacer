import cvxpy as cp
import numpy as np
from time import time
from cvxpy.atoms.affine.transpose import transpose

# Problem data.
m = 10
n = 5
np.random.seed(1)
A = np.random.randn(m, n)
b = np.random.randn(m)

# Construct the problem.
t = time()
x = cp.Variable(n)
#objective = cp.Minimize(cp.sum_squares(A @ x - b))
objective = cp.Minimize(cp.quad_form(x,np.eye(n)))
constraints = [0 <= x, x <= 1]
prob = cp.Problem(objective, constraints)

print("Optimal value", prob.solve())
print("Optimal var")
print(x.value) # A numpy ndarray.
print(time()-t)
