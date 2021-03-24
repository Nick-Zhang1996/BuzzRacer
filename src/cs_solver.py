import sys

from mosek.fusion import *
import numpy as np
import scipy.linalg
import scipy.stats
import time


class CSSolver:
    def __init__(self, n, m, l, N, u_min, u_max, mean_only=False, lti_k=False):
        try:
            M = Model()
            self.n = n
            self.m = m
            self.l = l
            self.N = N

            umax = np.tile(u_max.reshape((-1, 1)), (N, 1)).flatten()
            umin = np.tile(u_min.reshape((-1, 1)), (N, 1)).flatten()

            V = M.variable("V", m*N, Domain.inRange(umin, umax))
            k = None
            if not mean_only:
                if lti_k:
                    k = M.variable([m, n])
                    k_rep = Expr.repeat(k, N, 0)
                    k_rep2 = Expr.repeat(k_rep, N, 1)
                    identity_k = np.kron(np.eye(N), np.ones((m, n)))
                    K_dot = Matrix.sparse(identity_k)
                    K = Expr.mulElm(K_dot, k_rep2)
                else:
                    pattern = []
                    for ii in range(N):
                        for jj in range(m):
                            for ll in range(n):
                                row = ii*m + jj
                                col = ii*n + ll
                                pattern.append([row, col])
                    K = M.variable([m*N, n*N], Domain.sparse(Domain.unbounded(), pattern))
            else:
                pattern = []
                for ii in range(N):
                    for jj in range(m):
                        for ll in range(n):
                            row = ii * m + jj
                            col = ii * n + ll
                            pattern.append([row, col])
                K = M.parameter([m*N, n*N], pattern)

            # linear variables with quadratic cone constraints
            w = M.variable("w", 1, Domain.unbounded())
            x = M.variable("x", 1, Domain.unbounded())
            # y = M.variable("y", 1, Domain.unbounded())
            # z = M.variable("z", 1, Domain.unbounded())
            y1 = M.variable("y1", 1, Domain.unbounded())
            z1 = M.variable("z1", 1, Domain.unbounded())
            y2 = M.variable("y2", 1, Domain.unbounded())
            z2 = M.variable("z2", 1, Domain.unbounded())

            mu_0_T_A_T_Q_bar_B = M.parameter([1, m*N])
            vec_T_sigma_y_Q_bar_B = M.parameter([1, n*N*m*N])
            pattern = []
            for ii in range(n*N):
                for jj in range(m*N):
                    if jj <= ii:
                        pattern.append([ii, jj])
            pattern = []
            for ii in range(N):
                for jj in range(n):
                    for kk in range(m * (ii + 1)):
                        pattern.append([jj + n * ii, kk])
            Q_bar_half_B = M.parameter([n*N, m*N], pattern)
            pattern = []
            for ii in range(m*N):
                pattern.append([ii, ii])
            R_bar_half = M.parameter([m*N, m*N], pattern)
            pattern = []
            for ii in range(n * N):
                pattern.append([ii, ii])
            Q_bar_half = M.parameter([n*N, n*N], pattern)
            # sigma_y_half = M.parameter([n*N, n*N])
            A_sigma_0_half = M.parameter([n*N, n])
            pattern = []
            for ii in range(N):
                for jj in range(n):
                    for kk in range(l*(ii+1)):
                        pattern.append([jj + n * ii, kk])
            D = M.parameter([n*N, l*N], pattern)
            A_mu_0 = M.parameter([n*N, 1])
            pattern = []
            for ii in range(N):
                for jj in range(n):
                    for kk in range(m * (ii + 1)):
                        pattern.append([jj + n * ii, kk])
            B = M.parameter([n*N, m*N])
            d = M.parameter([n*N, 1])
            sigma_N_inv = M.parameter([n, n])
            neg_x_0_T_Q_B = M.parameter([1, m*N])
            d_T_Q_B = M.parameter([1, m*N])
            u_0 = M.parameter([m, 1])

            I = Matrix.eye(n*N)

            # convert to linear objective with quadratic cone constraints
            u = Expr.mul(mu_0_T_A_T_Q_bar_B, V)
            # coordinate shift, check to make sure T = *2
            q = Expr.mul(neg_x_0_T_Q_B, V)
            r = Expr.mul(d_T_Q_B, V)
            # v = Expr.mul(vec_T_sigma_y_Q_bar_B, Expr.flatten(K))
            if not mean_only:
                M.objective(ObjectiveSense.Minimize, Expr.add([q, r, u, w, x, y1, y2, z1, z2]))
                M.constraint(Expr.vstack(0.5, w, Expr.mul(Q_bar_half_B, V)), Domain.inRotatedQCone())
                M.constraint(Expr.vstack(0.5, x, Expr.mul(R_bar_half, V)), Domain.inRotatedQCone())

                M.constraint(Expr.vstack(0.5, y1, Expr.flatten(Expr.mul(Q_bar_half, Expr.mul(Expr.add(I, Expr.mul(B, K)), A_sigma_0_half)))), Domain.inRotatedQCone())
                # check BKD multiplicaion, or maybe with I? Something seems wrong here b/c sparsity pattern is invalid
                M.constraint(Expr.vstack(0.5, y2, Expr.flatten(Expr.mul(Q_bar_half, Expr.mul(Expr.add(I, Expr.mul(B, K)), D)))), Domain.inRotatedQCone())
                M.constraint(Expr.vstack(0.5, z1, Expr.flatten(Expr.mul(R_bar_half, Expr.mul(K, A_sigma_0_half)))), Domain.inRotatedQCone())
                M.constraint(Expr.vstack(0.5, z2, Expr.flatten(Expr.mul(R_bar_half, Expr.mul(K, D)))), Domain.inRotatedQCone())
            else:
                M.objective(ObjectiveSense.Minimize, Expr.add([q, r, u, w, x]))
                M.constraint(Expr.vstack(0.5, w, Expr.mul(Q_bar_half_B, V)), Domain.inRotatedQCone())
                M.constraint(Expr.vstack(0.5, x, Expr.mul(R_bar_half, V)), Domain.inRotatedQCone())

            M.constraint(Expr.sub(V.slice(2, N*m), V.slice(0, N*m-2)), Domain.inRange(-0.2, 0.2))
            u_oo = np.array([[0.0], [0.5]])
            # u_o = Matrix.dense(u_oo)
            self.u_o = M.parameter()
            self.u_o.setValue(0.3)
            self.u_s = M.parameter()
            u_0.setValue(0.5)
            # print(u_0.getValue())
            M.constraint(Expr.sub(self.u_o, V.index(1)), Domain.inRange(-0.2, 0.2))
            M.constraint(Expr.sub(self.u_s, V.index(0)), Domain.inRange(-0.2, 0.2))

            # M.constraint(K.slice([0, 0], [m, n]), Domain.equalsTo(K.slice([m, n], [2*m, 2*n])))

            # terminal mean constraint
            mu_N = np.zeros((n, 1))
            mu_N = np.array([7.5, 2., 2.5, 100., 100., 0.5, 1.0, 1000.]).reshape((8, 1))
            mu_N = Matrix.dense(mu_N)
            e_n = np.zeros((n, n))
            e_n[4, 4] = 1
            e_n[6, 6] = 1
            e_n = np.eye(n)
            E_N = Matrix.sparse(np.hstack((np.zeros((n, (N - 1) * n)), e_n)))
            E_N_T = Matrix.sparse(np.hstack((np.zeros((n, (N - 1) * n)), np.eye(n))).T)
            M.constraint(Expr.mul(E_N, Expr.add(Expr.add(A_mu_0, Expr.mul(B, V)), d)), Domain.lessThan(mu_N))
            e_n = -1 * e_n
            E_N = Matrix.sparse(np.hstack((np.zeros((n, (N - 1) * n)), e_n)))
            E_N_T = Matrix.sparse(np.hstack((np.zeros((n, (N - 1) * n)), np.eye(n))).T)
            M.constraint(Expr.mul(E_N, Expr.add(Expr.add(A_mu_0, Expr.mul(B, V)), d)), Domain.lessThan(mu_N))
            # terminal covariance constraint
            # sigma_0_part = M.variable()
            # M.constraint(Expr.vstack(sigma_0_part, Expr.flatten(
            #     Expr.mul(alpha_T, Expr.mul(E_k, Expr.mul(Expr.add(I, Expr.mul(B, K)), A_sigma_0_half))))),
            #              Domain.inQCone())
            # D_part = M.variable()
            # M.constraint(Expr.vstack(D_part, Expr.flatten(
            #     Expr.mul(alpha_T, Expr.mul(E_k, Expr.mul(Expr.add(I, Expr.mul(B, K)), D))))), Domain.inQCone())
            # cov_part = M.variable()
            # M.constraint(Expr.vstack(cov_part, sigma_0_part, D_part), Domain.inQCone())

            # chance constraint
            for ii in range(N):
                alpha = np.zeros((n, 1))
                alpha[6, 0] = 1
                alpha_T = Matrix.sparse(alpha.T)
                alpha = Matrix.sparse(alpha)
                beta = 2
                inv_prob = scipy.stats.norm.ppf(0.95)
                e_k = np.eye(n)
                E_k = Matrix.sparse(np.hstack((np.zeros((n, (ii) * n)), e_k, np.zeros((n, (N - ii - 1) * n)))))
                mean_part = Expr.mul(alpha_T, Expr.mul(E_k, Expr.add(Expr.add(A_mu_0, Expr.mul(B, V)), d)))
                # if not mean_only:
                sigma_0_part = M.variable()
                M.constraint(Expr.vstack(sigma_0_part, Expr.flatten(Expr.mul(alpha_T, Expr.mul(E_k, Expr.mul(Expr.add(I, Expr.mul(B, K)), A_sigma_0_half))))), Domain.inQCone())
                D_part = M.variable()
                M.constraint(Expr.vstack(D_part, Expr.flatten(Expr.mul(alpha_T, Expr.mul(E_k, Expr.mul(Expr.add(I, Expr.mul(B, K)), D)))).slice(0, (ii+1)*l)), Domain.inQCone())
                cov_part = M.variable()
                M.constraint(Expr.vstack(cov_part, sigma_0_part, D_part), Domain.inQCone())
                M.constraint(Expr.add(mean_part, Expr.mul(cov_part, inv_prob)), Domain.inRange(-beta, beta))
                # M.constraint(Expr.add(mean_part, Expr.mul(cov_part, inv_prob)), Domain.greaterThan(-beta))
                # else:
                #     M.constraint(Expr.add(mean_part, cov_part * inv_prob), Domain.lessThan(beta))
                # M.constraint(mean_part, Domain.lessThan(beta))

            # M.setLogHandler(sys.stdout)

            self.M = M
            self.V = V
            self.k = k
            self.K = K
            self.mu_0_T_A_T_Q_bar_B = mu_0_T_A_T_Q_bar_B
            self.vec_T_sigma_y_Q_bar_B = vec_T_sigma_y_Q_bar_B
            self.Q_bar_half_B = Q_bar_half_B
            self.R_bar_half = R_bar_half
            self.Q_bar_half = Q_bar_half
            # self.sigma_y_half = sigma_y_half
            self.A_sigma_0_half = A_sigma_0_half
            self.D = D
            self.A_mu_0 = A_mu_0
            self.B = B
            self.d = d
            self.sigma_N_inv = sigma_N_inv
            self.neg_x_0_T_Q_B = neg_x_0_T_Q_B
            self.d_T_Q_B = d_T_Q_B
            self.u_0 = u_0
            self.mean_only = mean_only
            self.lti_k = lti_k

        finally:
            pass
            # M.dispose()

    def populate_params(self, A, B, d, D, mu_0, sigma_0, sigma_N_inv, Q_bar, R_bar, u_0, x_target, K=None):
        n = 8
        m = 2
        l = 8
        N = self.N

        # A = np.tile(np.eye(n), (N, 1))
        # B = np.kron(np.eye(N), np.random.randn(n, m))
        # D = np.kron(np.eye(N), np.random.randn(n, l))
        # mu_0 = np.zeros((n, 1))
        # sigma_0 = np.random.randn(n, n)
        # sigma_0 = np.dot(sigma_0, sigma_0.T)
        # sigma_N_inv = np.linalg.inv(1000 * sigma_0)
        sigma_y = np.dot(A, np.dot(sigma_0, A.T)) + np.dot(D, D.T)
        # sigma_y = np.linalg.cholesky(sigma_y)
        # Q_bar = np.eye(n*N)
        # R_bar = np.eye(m*N)
        x_0 = x_target.copy()

        self.mu_0_T_A_T_Q_bar_B.setValue(2*np.dot(np.dot(np.dot(mu_0.T, A.T), Q_bar), B))
        temp = 2*np.dot(sigma_y, np.dot(Q_bar, B)).reshape((-1, 1)).T
        self.vec_T_sigma_y_Q_bar_B.setValue(temp)
        # try:
        #     self.Q_bar_half_B.setValue(np.dot(scipy.linalg.cholesky(Q_bar), B))
        # except np.linalg.LinAlgError:
        self.Q_bar_half_B.setValue(np.dot(np.sqrt(Q_bar), B))
        self.Q_bar_half.setValue(np.sqrt(Q_bar))
        # try:
        #     self.R_bar_half.setValue(scipy.linalg.cholesky(R_bar))
        # except np.linalg.LinAlgError:
        self.R_bar_half.setValue(np.sqrt(R_bar))
        # try:
        #     self.Q_bar_half.setValue(scipy.linalg.cholesky(Q_bar))
        # except np.linalg.LinAlgError:
        #     self.Q_bar_half.setValue(np.sqrt(Q_bar))
        # try:
        #     self.sigma_y_half.setValue(sigma_y)
        # except np.linalg.LinAlgError:
        #     print("cholesky failed")
        #     self.sigma_y_half.setValue(np.sqrt(sigma_y))
        # try:
        #     self.A_sigma_0_half.setValue(np.dot(A, np.linalg.cholesky(sigma_0)))
        # except np.linalg.LinAlgError:
        self.A_sigma_0_half.setValue(np.dot(A, np.sqrt(sigma_0)))
        self.D.setValue(D)
        self.A_mu_0.setValue((np.dot(A, mu_0)))
        self.B.setValue(B)
        self.d.setValue(d)
        self.sigma_N_inv.setValue(sigma_N_inv)
        self.neg_x_0_T_Q_B.setValue(2*np.dot(np.dot(-x_0.T, Q_bar), B))
        self.d_T_Q_B.setValue(2*np.dot(np.dot(d.T, Q_bar), B))
        u_oo = np.array([[0.1], [0.1]])
        self.u_o.setValue(u_0[1])
        self.u_s.setValue(u_0[0])
        # self.M.writeTask('dump.opf')
        if self.mean_only:
            self.K.setValue(K)

    def solve(self):
        # print(self.u_0.getValue())
        # self.M.solve()
        t0 = time.time()
        self.M.solve()
        print((time.time() - t0))
        try:
            if self.mean_only:
                K_level = np.zeros((self.m*self.N, self.n*self.N))
            else:
                if self.lti_k:
                    K_level = np.kron(np.eye(self.N), self.k.level().reshape((self.m, self.n)))
                else:
                    K_level = self.K.level()
            levels = (self.V.level(), K_level)
            # print(levels)
            return levels
        except SolutionError:
            raise RuntimeError

    def time(self):
        t0 = time.time()
        for ii in range(20):
            self.populate_params()
            self.solve()
        print((time.time() - t0) / 20)


def nearest_spd_cholesky(A):
    # print(np.linalg.eigvals(A))
    B = (A + A.T)/2
    U, Sigma, V = np.linalg.svd(B)
    H = np.dot(np.dot(V.T, np.diag(Sigma)), V)
    Ahat = (B+H)/2
    Ahat = (Ahat + Ahat.T)/2
    p = 1
    k = 0
    spacing = np.spacing(np.linalg.norm(A))
    I = np.eye(A.shape[0])
    while p != 0:
        k += 1
        try:
            R = np.linalg.cholesky(Ahat)
            p = 0
        except np.linalg.LinAlgError:
            eig = np.linalg.eigvals(Ahat)
            # print(eig)
            mineig = np.min(np.real(eig))
            print(mineig)
            Ahat = Ahat + I * (-mineig * k**2 + spacing)
    print(np.linalg.norm(Ahat - A))
    R_old = R.copy()
    R[np.abs(R) < 1e-5] = 1e-5
    np.tril(R)
    print(np.linalg.norm(R - R_old))
    return R


if __name__ == '__main__':
    solver = CSSolver()
    try:
        solver.populate_params()
        # solver.M.setSolverParam("numThreads", 8)
        # solver.M.setSolverParam("intpntCoTolPfeas", 1e-3)
        # solver.M.setSolverParam("intpntCoTolDfeas", 1e-3)
        # solver.M.setSolverParam("intpntCoTolRelGap", 1e-3)
        # solver.M.setSolverParam("intpntCoTolInfeas", 1e-3)
        solver.solve()
        # solver.time()
    finally:
        solver.M.dispose()
