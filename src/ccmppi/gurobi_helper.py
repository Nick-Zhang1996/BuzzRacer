# test construction of quadratic expression
import gurobipy as gp
from gurobipy import GRB
from gurobipy import *
import numpy as np

def vec(var):
    return vec.flatten('F')
def unvec(var,shape):
    return var.reshape(shape,order='F')

#untested
# express (tr(AX'Qx Qy Y B)) in vecX(m*n) and vecY(m*n)
# A: p*n
# Qx m*q
# Qy q*m
# B: n*p
def gurobi_trAXQYB(A,Qx,Qy,B,vecX,vecY):
    assert (A.shape[0] == B.shape[1])
    assert (A.shape[1] == B.shape[0])
    assert (Qx.shape[1] == Qy.shape[0])
    assert (Qx.shape[0] == Qy.shape[1])
    return vecX @ (np.kron(A, Qx.T).T @ (np.kron(B.T, Qy))) @ vecY


# express tr(AXB) in gurobi acceptable form
# A: p*m np matrix
# B: n*p np matrix
# vecX: vec(x), (m*n)*1 gurobi vector (1D)
# return: gurobi expression
def gurobi_trAXB(A,B,vecX):
    assert (A.shape[0] == B.shape[1])
    assert (vecX.shape[0] == A.shape[1]*B.shape[0])
    p = A.shape[0]
    m = A.shape[1]
    n = B.shape[0]
    # matrix.flatten('F') is vec(matrix)
    retval = (A.T.flatten('F').T @ np.kron(B.T,np.eye(m))) @ vecX
    #val = (A.T.flatten('F').T @ np.kron(B.T,np.eye(m)))
    #retval = sum( (val[i] * vecX[i]) for i in range(n*m))
    return retval

def gurobi_trAXB_alt(A,B,vecX):
    assert (A.shape[0] == B.shape[1])
    assert (vecX.shape[0] == A.shape[1]*B.shape[0])
    p = A.shape[0]
    m = A.shape[1]
    n = B.shape[0]
    retval = (np.eye(m).flatten('F').T @ np.kron( (B@A).T, np.eye(m) )) @ vecX
    #val = (np.eye(m).flatten('F').T @ np.kron( (B@A).T, np.eye(m) ))
    #retval = sum((val[i] * vecX[i]) for i in range(n*m))
    return retval

# express tr(A.T @ X.T @ Q @ X @ A)
# A: n*p, np matrix
# Q_sqrt: sqrt(Q), m*m, np matrix
# vecX: vec(X) X:m*n
# return: gurobi expression
def gurobi_matrix_quad(A, Q_sqrt, vecX):
    temp = np.kron(A.T, Q_sqrt)
    return vecX @ (temp.T @ temp) @ vecX

def test_bilinear():
    m = 2
    n = 3
    p = 4
    q = 5
    name = "quad"
    model = gp.Model(name)
    # supress output
    #model.setParam(GRB.Param.OutputFlag, 0)
    x_lb = np.random.rand(m*n)
    y_lb = np.random.rand(m*n)
    vecX = model.addMVar(shape=(m*n),lb=x_lb, ub=GRB.INFINITY, name='vecX')
    vecY = model.addMVar(shape=(m*n),lb=y_lb, ub=GRB.INFINITY, name='vecY')
    A = np.random.rand(p,n)
    Qx = np.random.rand(m,q)
    Qy = np.random.rand(q,m)
    B = np.random.rand(n,p)

    obj = gurobi_trAXQYB(A,Qx,Qy,B,vecX,vecY)
    model.setObjective(obj, GRB.MINIMIZE)
    model.optimize()

    # test numerival value
    vecX = vecX.x
    vecY = vecY.x
    X = unvec(vecX,(m,n))
    Y = unvec(vecY,(m,n))
    # ground truth
    err = gurobi_trAXQYB(A,Qx,Qy,B,vecX,vecY) - np.trace(A @ X.T @ Qx @ Qy @ Y @ B)
    print("error %.6f"%err)

def test_quad_fun():
    m = 2
    n = 3
    p = 3
    A = np.random.rand(n,p)
    # random symmetric matrix
    Q_sqrt = np.random.rand(m,m)
    Q_sqrt = Q_sqrt + Q_sqrt.T

    name = "quad"
    model = gp.Model(name)
    # supress output
    model.setParam(GRB.Param.OutputFlag, 0)
    vecX = model.addMVar(shape=(m*n), name='vecX')
    obj = gurobi_matrix_quad(A,Q_sqrt,vecX)
    model.addConstr( sum(vecX[i] for i in range(m*n)) == 1)
    model.setObjective(obj, GRB.MINIMIZE)
    model.optimize()
    print("decision variable")
    print(vecX.x)
    print("objective function")
    print(model.ObjVal)

    X = unvec(vecX.x,(m,n))
    obj_val = np.trace(A.T @ X.T @ Q_sqrt @ Q_sqrt @ X @ A)
    assert(model.ObjVal - obj_val < 0.0001)
    return



def test_AXB_fun(A,B,name,fun):
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

if __name__=="__1main__":
    m = 2
    n = 3
    p = 3
    A = np.random.rand(p,m)
    B = np.random.rand(n,p)
    print("A")
    print(A)
    print("B")
    print(B)
    test_AXB_fun(A,B,"1",gurobi_trAXB)
    test_AXB_fun(A,B,"2",gurobi_trAXB_alt)



if __name__=="__main__":
    #test_quad_fun()
    test_bilinear()

