# test construction of quadratic expression
import gurobipy as gp
from gurobipy import GRB
from gurobipy import *

import numpy as np

# express tr(AXB) in gurobi acceptable form
# A: p*m np matrix
# B: n*p np matrix
# vecX: vec(x), (m*n)*1 gurobi vector (1D)
def gurobi_trAXB(A,B,vecX):
    assert (A.shape[0] == B.shape[1])
    assert (vecX.shape[0] == A.shape[1]*B.shape[0])
    p = A.shape[0]
    m = A.shape[1]
    n = B.shape[0]
    # matrix.flatten('F') is vec(matrix)
    #retval = (A.T.flatten('F').T @ np.kron(B.T,np.eye(m))) @ vecX
    val = (A.T.flatten('F').T @ np.kron(B.T,np.eye(m)))
    retval = sum( (val[i] * vecX[i]) for i in range(n*m))
    return retval

def gurobi_trAXB_alt(A,B,vecX):
    assert (A.shape[0] == B.shape[1])
    assert (vecX.shape[0] == A.shape[1]*B.shape[0])
    p = A.shape[0]
    m = A.shape[1]
    n = B.shape[0]
    #retval = (np.eye(m).flatten('F').T @ np.kron( (B@A).T, np.eye(m) )) @ vecX
    val = (np.eye(m).flatten('F').T @ np.kron( (B@A).T, np.eye(m) ))
    retval = sum((val[i] * vecX[i]) for i in range(n*m))
    return retval

def test_fun(A,B,name,fun):
    p = A.shape[0]
    m = A.shape[1]
    n = B.shape[0]
    print('-------')
    print(name)
    model = gp.Model(name)
    # supress output
    model.setParam(GRB.Param.OutputFlag, 0)
    vecX = model.addMVar(shape=(m*n), name='vecX')
    obj = fun(A,B,vecX)
    model.addConstr( sum(vecX[i] for i in range(m*n)) == 1)

    model.setObjective(obj, GRB.MINIMIZE)
    model.optimize()
    print("decision variable")
    print(vecX.x)
    print("objective function")
    print(model.ObjVal)
    # re-calculate objective function value
    X = vecX.x.reshape((m,n),order='F')

    original = np.trace(A@X@B)
    form1 =  (A.T.flatten('F').T @ np.kron(B.T,np.eye(m))) @ X.flatten('F')
    form2 = (np.eye(m).flatten('F').T @ np.kron( (B@A).T, np.eye(m) )) @ X.flatten('F')
    # conform that the two forms are true to original
    assert(original - form1 < 0.001)
    assert(original - form2 < 0.001)

    form3 =  (A.T.flatten('F').T @ np.kron(B.T,np.eye(m))) @ vecX.x
    form4 = (np.eye(m).flatten('F').T @ np.kron( (B@A).T, np.eye(m) )) @ vecX.x
    assert(original - form3 < 0.001)
    assert(original - form4 < 0.001)


    print("objective function real")
    print(original)
    print('-------')

if __name__=="__main__":
    m = 2
    n = 3
    p = 3
    A = np.random.rand(p,m)
    B = np.random.rand(n,p)
    print("A")
    print(A)
    print("B")
    print(B)
    test_fun(A,B,"1",gurobi_trAXB)
    test_fun(A,B,"2",gurobi_trAXB_alt)


