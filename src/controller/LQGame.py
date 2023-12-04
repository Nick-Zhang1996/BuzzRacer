import numpy as np
from util.timeUtil import execution_timer

def my_solve_lq_game(As, Bs, Qs, qs, Rs, rs, t):
    '''
    solve a linear quadratic game defined as follows:
    notations:
    x+ := x_{k+1}, Q+ := Q_{k+1}
    Ai_k -> the index directly following a term denotes an agent (i,j), 
       the index after underscore(_) denotes time (k)

    Objective function
    Ji = sum_{k=0}^{K} gi_k
    NOTE: omitted _k for clarity, uj = uj_k, Rj = Rj_k
    gi_k = 1/2 x+.T @ Q+ @ x+ + q+.T @ x+ + 1/2 sum_j [uj.T @ Rj @ uj.T + 2 rj.T @ uj]
    x+ = Ak @ x + sum (Bj_k @ uj_k)
    As = [A_0, A_1, ... A_K]
    Bs = [B1s,B2s,... Bis,..BNs], for each of the N agents. 
       B0s = [B0_0, B0_1,...B0_K] B for agent 0 at each time step k
    Qs = [Q1s, Q2s, ..., QNs] for each of the N agents
        NOTE: the index starts from 1
       Qis = [Qi_1, Qi_2, ..., Qi_K+1]
    qs = [q1, q2, ... Q
        NOTE: the index starts from 1
       qis = [qi_1, qi_2, ..., qi_K+1]
    Rs = [R1s, R2s, ..., RNs]
        Ris = [Ri_0, Ri_1, ..., Ri_K]
    rs = [r1s, r2s, ..., rNs]
        ris = [ri_0, ri_1, ..., ri_K]

    optimal control
    ui_k = -Pi_k @ x_k + vi_k
    [return]: Ps,vs
    Ps = [P1s,P2s, ... PNs]
      Pis = [Pi_0, Pi_1, .. Pi_K]
    vs = [v1s,v2s, ... vNs]
      vis = [vi_0, vi_1, .. vi_K]
    '''
    t.s()
    t.s('setup')
    K = len(As)-1
    N = len(Bs)
    n = As[0].shape[0]
    m = Bs[0][0].shape[1]
    assert(As[0].shape == (n,n))
    assert(Bs[0][0].shape == (n,m))
    assert(Qs[0][0].shape == (n,n))
    assert(qs[0][0].shape == (n,1))
    assert(Rs[0][0].shape == (m,m))
    assert(rs[0][0].shape == (m,1))
    assert(len(Bs) == N)
    assert(len(Bs[0]) == (K+1))
    assert(len(Qs) == N)
    assert(len(Qs[0]) == (K+1))
    assert(len(qs) == N)
    assert(len(qs[0]) == (K+1))
    assert(len(Rs) == N)
    assert(len(Rs[0]) == (K+1))
    assert(len(rs) == N)
    assert(len(rs[0]) == (K+1))
    A = As
    B = Bs
    Q = Qs
    q = qs
    R = Rs
    r = rs

    # value function / min cost to go
    # Vk(x) = 1/2 x.T @ Q_bar_k x + L_bar_k @ x + C_bar_k

    # corresponds to Q_bar_{k+1} in derivation
    Q_bar = [np.zeros((n,n)) for i in range(N)]
    L_bar = [np.zeros((1,n)) for i in range(N)]
    C_bar = [np.zeros((1,1)) for i in range(N)]

    vs = []
    Ps = []
    t.e('setup')

    for k in range(K,-1,-1):
        # v = [v1_k, v2_k, ..., vN_k] (N*m * 1 vector)
        # Av @ v = bv
        # construct Av, bv
        t.s('Avbv')
        Av = np.zeros((N*m,N*m))
        bv = np.zeros((N*m,1))
        for i in range(0,N):
            # Q,q corresbonds to Q+, q+ in derivation, due to index starting from 1
            # we use Q[i][k] for Q+_i
            # Av_i @ v_i + sum Av_j @ v_j = bv_i
            Av_i = R[i][k] + B[i][k].T @ Q[i][k] @ B[i][k] + B[i][k].T @ Q_bar[i] @ B[i][k]
            Av[i*m:(i+1)*m,i*m:(i+1)*m] = Av_i
            bv_i = - ( B[i][k].T @ q[i][k] + r[i][k] + B[i][k].T @ L_bar[i].T )
            bv[i*m:(i+1)*m] = bv_i
            for j in range(0,N):
                if (j==i):
                    continue
                Av_j = B[i][k].T @ (Q[i][k] + Q_bar[i]) @ B[j][k]
                Av[i*m:(i+1)*m,j*m:(j+1)*m] = Av_j
            #TODO check that only the first i*m rows are non-zero
        t.e('Avbv')

        # P = np.vstack([P1_k, P2_k, .. PN_k]) (m*N * n matrix)
        # Ap @ P = bp
        # NOTE Ap = -Av
        t.s('bp')
        #Ap = np.zeros((m*N,m*N))
        Ap = -Av
        bp = np.zeros((m*N,n))
        for i in range(0,N):
            #Ap_i = -( R[i][k] + B[i][k].T @ Q[i][k] @ B[i][k] + B[i][k].T @ Q_bar[i] @ B[i][k] )
            #Ap[i*m:(i+1)*m,i*m:(i+1)*m] = Ap_i
            bp_i = - ( B[i][k].T @ (Q[i][k]+Q_bar[i]) @ A[k] )
            bp[i*m:(i+1)*m] = bp_i
            for j in range(0,N):
                if (j==i):
                    continue
                #Ap_j = -(B[i][k].T @ (Q_bar[i]+Q[i][k]) @ B[j][k])
                #Ap[i*m:(i+1)*m,j*m:(j+1)*m] = Ap_j
            #TODO check that only the first i*m rows are non-zero
        t.e('bp')
        t.s('lsqsq')
        v_k, residuals_v, rank_v, s_v = np.linalg.lstsq(a=Av, b=bv, rcond=None)
        P_k, residuals_p, rank_p, s_p = np.linalg.lstsq(a=Ap, b=bp, rcond=None)
        t.e('lsqsq')

        v_k = v_k.reshape(N,m)
        P_k = P_k.reshape(N,m,n)

        vs.append(v_k)
        Ps.append(P_k)

        t.s('value fun')
        Fk = (A[k] - sum([B[j][k] @ P_k[j] for j in range(N)]))
        Mk = sum([B[j][k] @ v_k[j] for j in range(N)])
        # update Q_bar, L_bar,C_bar for backpropagate value function
        temp_Q = sum([P_k[j].T @ R[j][k] @ P_k[j] for j in range(N)]) 
        temp_L = sum([v_k[j].T @ R[j][k] @ P_k[j] + r[j][k].T @ P_k[j] for j in range(N)])
        for i in range(0,N):
            # value due to V(k+1,x+)
            Qv = Fk.T @ Q_bar[i] @ Fk
            Lv = Mk.T @ Q_bar[i] @ Fk + L_bar[i] @ Fk
            #Cv = 0.5*Mk.T @ Q_bar[i] @ Mk + L_bar[i] @ Mk + C_bar[i]
            #Cv = Cv.reshape(1,1)

            # value due to g(k,x+,u)
            #Qg = Fk.T @ Q[i][k] @ Fk + sum([P_k[j].T @ R[j][k] @ P_k[j] for j in range(N)]) 
            #Lg = Mk.T @ Q[i][k] @ Fk + q[i][k].T @ Fk - sum([v_k[j].T @ R[j][k] @ P_k[j] + r[j][k].T @ P_k[j] for j in range(N)])
            #Cg = 0.5*Mk.T @ Q[i][k] @ Mk + q[i][k].T @ Mk + 0.5*sum([v_k[j].T @ R[j][k] @ v_k[j] + 2*r[j][k].T @ v_k[j] for j in range(N)])
            #Cg = Cg.reshape(1,1)

            Qg = Fk.T @ Q[i][k] @ Fk + temp_Q
            Lg = Mk.T @ Q[i][k] @ Fk + q[i][k].T @ Fk - temp_L

            Q_bar[i] = Qv + Qg
            L_bar[i] = Lv + Lg
            #C_bar[i] = Cv + Cg

            assert(Q_bar[i].shape == (n,n))
            assert(L_bar[i].shape == (1,n))
            assert(C_bar[i].shape == (1,1))
        t.e('value fun')

    t.s('reorder')
    # TODO reorder vs
    # before: vs[K-k][i] => vi_k
    # after: vs[i][k] => vi_k
    new_vs = []
    new_Ps = []
    for i in range(0,N):
        v_i = []
        P_i = []
        for k in range(K+1):
            v_i.append(vs[K-k][i].reshape(m,1))
            P_i.append(Ps[K-k][i])
        new_vs.append(v_i)
        new_Ps.append(P_i)
    t.e('reorder')
    t.e()
    return (np.array(new_Ps), np.array(new_vs))
