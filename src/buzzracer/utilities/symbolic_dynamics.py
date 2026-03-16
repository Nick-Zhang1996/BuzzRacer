from sympy import symbols, sin, cos, diff
import numpy as np


class SymbolicDynamics:
    """helper for calculation jacobian and hessians in a control problem After
    instantiating class, user should define self.f as a function of self.x,
    self.u."""

    def __init__(self, n, m):
        self.n = n
        self.m = m
        self.x = [symbols(f'x{i}') for i in range(n)]
        self.u = [symbols(f'u{i}') for i in range(m)]
        self.dfdx = None
        self.dfdu = None

        # to be set by user
        self.f = None
        self.l = None
        return

    def sym_der(self):
        ''' calculate jacobian of f(x,u)
        before calling this function user should define
        self.f as a function of self.x, self.u
        self.f: list(symbols) -> [f0,f1,f2,...,fn], functions of self.x,self.u
        self.dfdx: nested list of size (n,n), dfdx[i,j] = dfi_dxj
        '''
        dfdx = []
        n = self.n
        f = self.f
        x = self.x
        u = self.u
        assert (len(f) == n)
        for i in range(n):
            dfi_dx = []
            for j in range(n):
                dfi_dx.append(f[i].diff(x[j]))
            dfdx.append(dfi_dx)
        self.dfdx = dfdx

        dfdu = []
        m = self.m
        for i in range(n):
            dfi_du = []
            for j in range(m):
                dfi_du.append(f[i].diff(u[j]))
            dfdu.append(dfi_du)
        self.dfdu = dfdu
        '''
        # 1*n
        self.lx = []
        # n*n
        self.lxx = []
        for i in range(n):
            dl_dx_dx = []
            self.lx.append(self.l.diff(x[i]))
            for j in range(n):
                dl_dx_dx.append(self.lx[-1].diff(x[j]))
            self.lxx.append(dl_dx_dx)

        # 1*m
        self.lu = []
        # m*m
        self.luu = []
        # m*n
        self.lux = []
        for i in range(m):
            dl_du_du = []
            dl_du_dx = []
            self.lu.append(self.l.diff(u[i]))
            for j in range(n):
                dl_du_dx.append(self.lu[-1].diff(x[j]))
            for j in range(m):
                dl_du_du.append(self.lu[-1].diff(u[j]))
            self.lux.append(dl_du_dx)
            self.luu.append(dl_du_du)
        '''
        return

    def calc_der(self, x0, u0, subs_dict={}):
        ''' calculate fx,fu,lx,lu,lxx,luu,lux numerically
        e.g. fx = dfdx = df(x,u) / dx 
        user should call sym_der() before calling this function
        x0: list-like, substitute values for x
        u0: list-like, substitute values for u
        return: dfdx,dudx in numpy array, evaluated at x0,u0
        '''
        x0 = np.array(x0).flatten()
        u0 = np.array(u0).flatten()
        assert (len(x0) == self.n)
        assert (len(u0) == self.m)
        for i in range(self.n):
            subs_dict.update({self.x[i]: x0[i]})
        for i in range(self.m):
            subs_dict.update({self.u[i]: u0[i]})
        dfdx_val = np.array([[eq.evalf(subs=subs_dict) for eq in row]
                            for row in self.dfdx], dtype=np.float64)
        dfdu_val = np.array([[eq.evalf(subs=subs_dict) for eq in row]
                            for row in self.dfdu], dtype=np.float64)

        '''
        dldx_val = np.array([eq.evalf(subs=subs_dict) for eq in self.lx])
        dldu_val = np.array([eq.evalf(subs=subs_dict) for eq in self.lu])
        dldxdx_val = np.array([[eq.evalf(subs=subs_dict) for eq in row] for row in self.lxx])
        dldudu_val = np.array([[eq.evalf(subs=subs_dict) for eq in row] for row in self.luu])
        dldudx_val = np.array([[eq.evalf(subs=subs_dict) for eq in row] for row in self.lux])
        return dfdx_val, dfdu_val, dldx_val, dldu_val, dldxdx_val, dldudu_val,dldudx_val
        '''
        return dfdx_val, dfdu_val

    def xQx_diag(self, x, Q):
        ''' calculate x.T @ Q @ x, with x being a vector of symbolic variables
        x = [x0,x1,...] dim n
        Q = np.array, dim n*n
        only consider diagonal terms
        '''
        assert (len(x) == Q.shape[0])
        assert (Q.shape[0] == Q.shape[1])
        result = 0
        for i in range(x):
            result += x[i]*x[i]*Q[i, i]
        return result

    def product(self, a, b):
        """calculate inner of two vectors."""
        assert (len(a) == len(b))
        result = 0
        for i in range(len(a)):
            result += a[i]*b[i]
        return result

    def minus(self, a, b):
        assert (len(a) == len(b))
        return [a[i]-b[i] for i in range(len(a))]


if __name__ == '__main__':
    test = SymbolicDynamics(3, 2)
    x0 = test.x[0]
    x1 = test.x[1]
    x2 = test.x[2]
    u0 = test.u[0]
    u1 = test.u[1]

    f0 = x0 + x0*x2 + x2*u0 + x0**2*u1
    f1 = x1*cos(u0)
    f2 = (10+x0)*u1**2
    test.f = [f0, f1, f2]
    test.l = x0+x1+x2+u0+u1*x0

    test.sym_der()
    fx, fu, lx, lu, lxx, luu, lux = test.calc_der(x0=[1, 2, 3], u0=[4, 5])
    print(lxx)
    print(luu)
    print(lux)
