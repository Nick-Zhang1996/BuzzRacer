import time
import numpy as np
from numpy import sin, cos, tan, arctan as atan, sqrt, arctan2 as atan2, zeros, zeros_like, abs, pi
import scipy.io
import matplotlib.pyplot as plt
import torch
import throttle_model
from multiprocessing import Process
from multiprocessing.dummy import DummyProcess


class Model:
    def __init__(self, N):
        self.throttle = throttle_model.Net()
        self.throttle.load_state_dict(torch.load('throttle_model1.pth'))
        self.N = N

    def get_curvature(self, s):
        rho = np.zeros_like(s)
        map_params = [
                [2.78, -2.97, -0.6613, 0, 3.8022, 0],
                [10.04, 6.19, 2.4829, 3.8022, 18.3537, 0.1712],
                [1.46, 13.11, 2.4829, 22.1559, 11.0228, 0],
                [-5.92, 3.80, -0.6613, 33.1787, 18.6666, 0.1683],
                [-0.24, -0.66, -0.6613, 51.8453, 7.2218, 0]
            ]
        num_segments = 5
        while (s > map_params[num_segments-1][3] + map_params[num_segments-1][4]).any():
            s[s > map_params[num_segments-1][3] + map_params[num_segments-1][4]] -= map_params[num_segments-1][3] + map_params[num_segments-1][4]
        for ii in range(num_segments):
            truths = np.where(np.logical_and(map_params[ii][3] <= s, s <= map_params[ii][3] + map_params[ii][4]))
            rho[truths] = map_params[ii][5]
        return rho

    def update_dynamics(self, state, input, dt, nn=None, throttle_nn=None, cartesian=np.array([])):
        state = state.T
        input = input.T
        m_Vehicle_m = 21.7562#1270
        m_Vehicle_Iz = 1.124#2000
        m_Vehicle_lF = 0.34#1.015
        lFR = 0.57#3.02
        m_Vehicle_lR = lFR-m_Vehicle_lF
        m_Vehicle_IwF = 0.1#8
        m_Vehicle_IwR = .0373
        m_Vehicle_rF = 0.095#0.325
        m_Vehicle_rR = 0.090#0.325
        m_Vehicle_h = 0.12#.54
        m_g = 9.80665

        tire_B = 4.0#10
        tire_C = 1.0
        tire_D = 1.0
        tire_E = 1.0
        tire_Sh = 0.0
        tire_Sv = 0.0

        N, dx = state.shape
        m_nu = 1

        vx = state[:, 0]
        vy = state[:, 1]
        wz = state[:, 2]
        wF = state[:, 3]
        wR = state[:, 4]
        psi = state[:, 5]
        X = state[:, 6]
        Y = state[:, 7]

        if (vx < 0.1).any():
            vx = np.ones_like(vx) * 0.1
        if (wF < 1).any():
            wF = np.ones_like(wF) * 1
        if (wR < 1).any():
            wR = np.ones_like(wR) * 1


        m_Vehicle_kSteering = -0.24 # -pi / 180 * 18.7861
        m_Vehicle_cSteering = -0.02 # 0.0109
        throttle_factor = 0.38
        # delta = input[:, 0]
        steering = input[:, 0]
        delta = m_Vehicle_kSteering * steering + m_Vehicle_cSteering
        T = np.maximum(input[:, 1], 0)

        min_velo = 0.1
        deltaT = 0.01
        t = 0

        while t < dt:
            beta = atan2(vy, vx)

            V = sqrt(vx * vx + vy * vy)
            vFx = V * cos(beta - delta) + wz * m_Vehicle_lF * sin(delta)
            vFy = V * sin(beta - delta) + wz * m_Vehicle_lF * cos(delta)
            vRx = vx
            vRy = vy - wz * m_Vehicle_lR

            sEF = -(vFx - wF * m_Vehicle_rF) / (vFx) + tire_Sh
            muFx = tire_D * sin(tire_C * atan(tire_B * sEF - tire_E * (tire_B * sEF - atan(tire_B * sEF)))) + tire_Sv
            sEF = -(vRx - wR * m_Vehicle_rR) / (vRx) + tire_Sh
            muRx = tire_D * sin(tire_C * atan(tire_B * sEF - tire_E * (tire_B * sEF - atan(tire_B * sEF)))) + tire_Sv

            sEF = atan(vFy / abs(vFx)) + tire_Sh
            alpha = -sEF
            muFy = -tire_D * sin(tire_C * atan(tire_B * sEF - tire_E * (tire_B * sEF - atan(tire_B * sEF)))) + tire_Sv
            sEF = atan(vRy / abs(vRx)) + tire_Sh
            alphaR = -sEF
            muRy = -tire_D * sin(tire_C * atan(tire_B * sEF - tire_E * (tire_B * sEF - atan(tire_B * sEF)))) + tire_Sv

            fFz = m_Vehicle_m * m_g * (m_Vehicle_lR - m_Vehicle_h * muRx) / (
                    m_Vehicle_lF + m_Vehicle_lR + m_Vehicle_h * (muFx * cos(delta) - muFy * sin(delta) - muRx))
            # fFz = m_Vehicle_m * m_g * (m_Vehicle_lR / 0.57)
            fRz = m_Vehicle_m * m_g - fFz

            fFx = fFz * muFx
            fRx = fRz * muRx
            fFy = fFz * muFy
            fRy = fRz * muRy

            ax = ((fFx * cos(delta) - fFy * sin(delta) + fRx) / m_Vehicle_m + vy * wz)



            next_state = zeros_like(state)
            next_state[:, 0] = vx + deltaT * ((fFx * cos(delta) - fFy * sin(delta) + fRx) / m_Vehicle_m + vy * wz)
            next_state[:, 1] = vy + deltaT * ((fFx * sin(delta) + fFy * cos(delta) + fRy) / m_Vehicle_m - vx * wz)
            vy_dot = ((fFx * sin(delta) + fFy * cos(delta) + fRy) / m_Vehicle_m - vx * wz)
            if nn:
                pass
                # input_tensor = torch.from_numpy(np.vstack((steering, vx, vy, wz, ax, wF, wR)).T).float()
                # # input_tensor = torch.from_numpy(input).float()
                # forces = nn(input_tensor).detach().numpy()
                # fafy = forces[:, 0]
                # fary = forces[:, 1]
                # fafx= forces[0, 2]
                # farx = forces[0, 3]
                #
                # next_state[:, 0] = vx + deltaT * ((fafx + farx) / m_Vehicle_m + vy * wz)
                # next_state[:, 1] = vy + deltaT * ((fafy + fary) / m_Vehicle_m - vx * wz)
                # next_state[:, 2] = wz + deltaT * ((fafy) * m_Vehicle_lF - fary * m_Vehicle_lR) / m_Vehicle_Iz
            else:
                next_state[:, 1] = vy + deltaT * ((fFx * sin(delta) + fFy * cos(delta) + fRy) / m_Vehicle_m - vx * wz)
                next_state[:, 2] = wz + deltaT * (
                            (fFy * cos(delta) + fFx * sin(delta)) * m_Vehicle_lF - fRy * m_Vehicle_lR) / m_Vehicle_Iz
            next_state[:, 3] = wF - deltaT * m_Vehicle_rF / m_Vehicle_IwF * fFx
            if throttle_nn:
                input_tensor = torch.from_numpy(np.hstack((T.reshape((-1, 1)), wR.reshape((-1, 1)) / throttle_factor))).float()
                next_state[:, 4] = wR + deltaT * throttle_nn(input_tensor).detach().numpy().flatten()
            else:
                next_state[:, 4] = T  # wR + deltaT * (m_Vehicle_kTorque * (T-wR) - m_Vehicle_rR * fRx) / m_Vehicle_IwR
            rho = self.get_curvature(Y)
            next_state[:, 5] = psi + deltaT * (wz - (vx * cos(psi) - vy * sin(psi)) / (1 - rho * X) * rho)
            next_state[:, 6] = X + deltaT * (vx * sin(psi) + vy * cos(psi))
            next_state[:, 7] = Y + deltaT * (vx * cos(psi) - vy * sin(psi)) / (1 - rho * X)

            if len(cartesian) > 0:
                cartesian[0, :] += deltaT * wz
                cartesian[1, :] += deltaT * (cos(cartesian[0, :]) * vx - sin(cartesian[0, :]) * vy)
                cartesian[2, :] += deltaT * (sin(cartesian[0, :]) * vx + cos(cartesian[0, :]) * vy)

            t += deltaT
            vx = next_state[:, 0]
            vy = next_state[:, 1]
            wz = next_state[:, 2]
            wF = next_state[:, 3]
            wR = next_state[:, 4]
            psi = next_state[:, 5]
            X = next_state[:, 6]
            Y = next_state[:, 7]

        if len(cartesian) > 0:
            return next_state.T, cartesian
        else:
            return next_state.T

    def linearize_dynamics(self, states, controls):
        nx = 8
        nu = 2
        nN = self.N
        dt = 0.1

        delta_x = np.array([0.01, 0.001, 0.01, 0.1, 0.1, 0.05, 0.1, 0.2])
        delta_u = np.array([0.001, 0.01])
        delta_x_flat = np.tile(delta_x, (1, nN))
        delta_u_flat = np.tile(delta_u, (1, nN))
        delta_x_final = np.multiply(np.tile(np.eye(nx), (1, nN)), delta_x_flat)
        delta_u_final = np.multiply(np.tile(np.eye(nu), (1, nN)), delta_u_flat)
        xx = np.tile(states, (nx, 1)).reshape((nx, nx*nN), order='F')
        # print(delta_x_final, xx)
        ux = np.tile(controls, (nx, 1)).reshape((nu, nx*nN), order='F')
        x_plus = xx + delta_x_final
        # print(x_plus, ux)
        x_minus = xx - delta_x_final
        fx_plus = self.update_dynamics(x_plus, ux, dt, throttle_nn=self.throttle)
        # print(fx_plus)
        fx_minus = self.update_dynamics(x_minus, ux, dt, throttle_nn=self.throttle)
        A = (fx_plus - fx_minus) / (2 * delta_x_flat)

        xu = np.tile(states, (nu, 1)).reshape((nx, nu*nN), order='F')
        uu = np.tile(controls, (nu, 1)).reshape((nu, nu*nN), order='F')
        u_plus = uu + delta_u_final
        # print(xu)
        u_minus = uu - delta_u_final
        fu_plus = self.update_dynamics(xu, u_plus, dt, throttle_nn=self.throttle)
        # print(fu_plus)
        fu_minus = self.update_dynamics(xu, u_minus, dt, throttle_nn=self.throttle)
        B = (fu_plus - fu_minus) / (2 * delta_u_flat)

        state_row = np.zeros((nx*nN, nN))
        input_row = np.zeros((nu*nN, nN))
        for ii in range(nN):
            state_row[ii*nx:ii*nx + nx, ii] = states[:, ii]
            input_row[ii*nu:ii*nu+nu, ii] = controls[:, ii]
        d = self.update_dynamics(states, controls, dt, throttle_nn=self.throttle) - np.dot(A, state_row) - np.dot(B, input_row)

        return A, B, d

    def form_long_matrices_LTI(self, A, B, D):
        nx = 8
        nu = 2
        N = self.N

        AA = np.zeros((nx*N, nx))
        BB = zeros((nx*N, nu * N))
        DD = zeros((nx, nx * N))
        B_i_row = zeros((nx, 0))
        # D_i_bar = zeros((nx, nx))
        for ii in np.arange(0, N):
            AA[ii*nx:(ii+1)*nx, :] = np.linalg.matrix_power(A, ii+1)

            B_i_cell = np.dot(np.linalg.matrix_power(A, ii), B)
            B_i_row = np.hstack((B_i_cell, B_i_row))
            BB[ii*nx:(ii+1)*nx, :(ii+1)*nu] = B_i_row

            # D_i_bar = np.hstack((np.dot(np.linalg.matrix_power(A, ii - 1), D), D_i_bar))
            # temp = np.hstack((D_i_bar, np.zeros((nx, max(0, nx * N - D_i_bar.shape[1])))))
            # DD = np.vstack((DD, temp[:, 0: nx * N]))

        return AA, BB, DD

    def form_long_matrices_LTV(self, A, B, d, D):
        nx = 8
        nu = 2
        nl = 8
        N = self.N

        AA = np.zeros((nx*N, nx))
        BB = zeros((nx*N, nu * N))
        dd = zeros((nx*N, 1))
        DD = zeros((nx*N, nl * N))
        AA_i_row = np.eye(nx)
        dd_i_row = np.zeros((nx, 1))
        # B_i_row = zeros((nx, 0))
        # D_i_bar = zeros((nx, nx))
        for ii in np.arange(0, N):
            AA_i_row = np.dot(A[:, :, ii], AA_i_row)
            AA[ii*nx:(ii+1)*nx, :] = AA_i_row

            B_i_row = B[:, :, ii]
            D_i_row = D[:, :, ii]
            for jj in np.arange(ii-1, -1, -1):
                B_i_cell = np.dot(A[:, :, ii], BB[(ii-1)*nx:ii*nx, jj*nu:(jj+1)*nu])
                B_i_row = np.hstack((B_i_cell, B_i_row))
                D_i_cell = np.dot(A[:, :, ii], DD[(ii-1)*nx:ii*nx, jj*nl:(jj+1)*nl])
                D_i_row = np.hstack((D_i_cell, D_i_row))
            BB[ii*nx:(ii+1)*nx, :(ii+1)*nu] = B_i_row
            DD[ii*nx:(ii+1)*nx, :(ii+1)*nl] = D_i_row

            dd_i_row = np.dot(A[:, :, ii], dd_i_row) + d[:, :, ii]
            dd[ii*nx:(ii+1)*nx, :] = dd_i_row

        return AA, BB, dd, DD


def add_labels():
    plt.subplot(7, 2, 1)
    plt.gca().legend(('vx',))
    plt.xlabel('t (s)')
    plt.ylabel('m/s')
    plt.subplot(7, 2, 2)
    plt.gca().legend(('vy',))
    plt.xlabel('t (s)')
    plt.ylabel('m/s')
    plt.subplot(7, 2, 3)
    plt.gca().legend(('wz',))
    plt.xlabel('t (s)')
    plt.ylabel('rad/s')
    plt.subplot(7, 2, 4)
    plt.gca().legend(('wF',))
    plt.xlabel('t (s)')
    plt.ylabel('rad/s')
    plt.subplot(7, 2, 5)
    plt.gca().legend(('wR',))
    plt.xlabel('t (s)')
    plt.ylabel('rad/s')
    plt.subplot(7, 2, 6)
    plt.gca().legend(('e_psi',))
    plt.xlabel('t (s)')
    plt.ylabel('rad')
    plt.subplot(7, 2, 7)
    plt.gca().legend(('e_y',))
    plt.xlabel('t (s)')
    plt.ylabel('m')
    plt.subplot(7, 2, 8)
    plt.gca().legend(('s',))
    plt.xlabel('t (s)')
    plt.ylabel('m')
    plt.subplot(7, 2, 9)
    plt.gca().legend(('Yaw',))
    plt.xlabel('t (s)')
    plt.ylabel('rad')
    plt.subplot(7, 2, 10)
    plt.gca().legend(('X',))
    plt.xlabel('t (s)')
    plt.ylabel('m')
    plt.subplot(7, 2, 11)
    plt.gca().legend(('Y',))
    plt.xlabel('t (s)')
    plt.ylabel('m')
    plt.subplot(7, 2, 12)
    # plt.gca().legend(('s',))
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.subplot(7, 2, 13)
    plt.gca().legend(('steering',))
    plt.xlabel('t (s)')
    # plt.ylabel('')
    plt.subplot(7, 2, 14)
    plt.gca().legend(('throttle',))
    plt.xlabel('t (s)')
    # plt.ylabel('m')


def plot(states, controls, sim_length):
    plt.figure()
    time = np.arange(sim_length) / 10
    mat = scipy.io.loadmat("mppi_data/track_boundaries.mat")
    inner = mat['track_inner'].T
    outer = mat['track_outer'].T
    for ii in range(14):
        plt.subplot(7, 2, ii + 1)
        if ii < 11:
            plt.plot(time, states[ii, :])
        elif ii == 11:
            plt.plot(states[-2, :], states[-1, :])
            plt.plot(inner[0, :], inner[1, :], 'k')
            plt.plot(outer[0, :], outer[1, :], 'k')
        else:
            plt.plot(time, controls[ii-12, :])
    # states = np.load('cs_2Hz_states.npz.npy')
    # controls = np.load('cs_2Hz_control.npz.npy')
    # for ii in range(14):
    #     plt.subplot(7, 2, ii + 1)
    #     if ii < 11:
    #         plt.plot(time, states[ii, :])
    #     elif ii == 11:
    #         plt.plot(states[-2, :], states[-1, :])
    #         plt.plot(inner[0, :], inner[1, :], 'k')
    #         plt.plot(outer[0, :], outer[1, :], 'k')
    #     else:
    #         plt.plot(time, controls[ii-12, :])
    add_labels()
    # mat = scipy.io.loadmat('mppi_data/track_boundaries.mat')
    # inner = mat['track_inner'].T
    # outer = mat['track_outer'].T
    # plt.subplot(7, 2, 12)
    # plt.plot(inner[0, :], inner[1, :], 'k')
    # plt.plot(outer[0, :], outer[1, :], 'k')
    # np.save('cs_5Hz_states', states)
    # np.save('cs_5Hz_control', controls)
    plt.show()
    plt.figure()
    plt.plot(states[-2, :], states[-1, :])
    # states = np.load('cs_5Hz_states.npy')
    # plt.plot(states[-2, :], states[-1, :])
    plt.plot(inner[0, :], inner[1, :], 'k')
    plt.plot(outer[0, :], outer[1, :], 'k')
    # plt.gca().legend(('ltv mpc', 'cs smpc', 'track boundaries'))
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.show()


def run_simple_controller():
    n = 8
    m = 2
    l = 8
    N = 10
    ar = Model(N)
    x_target = np.tile(np.array([7, 0, 0, 0, 0, 0, 0, 0]).reshape((-1, 1)), (N, 1))
    x = np.array([4., 0., 0., 50., 50., 0.1, 0., 0.]).reshape((8, 1))
    state = x.copy()
    cartesian = np.array([-0.6613+0.1, 2.78-3.25, -2.97+2.3]).reshape((-1, 1))
    y = state.copy()
    u = np.array([0.01, 0.5]).reshape((2, 1))
    xs = np.tile(x, (1, N))
    us = np.tile(u, (1, N))
    u_min = np.array([-0.9, -0.9])
    u_max = np.array([0.9, 0.9])
    sigma_0 = np.zeros((n, n))
    # sigma_0 = np.random.randn(n, n)
    # sigma_0 = np.dot(sigma_0, sigma_0.T)
    # sigma_0 = np.eye(n)
    sigma_N_inv = sigma_0
    Q = np.zeros((n, n))
    Q[0, 0] = 3
    Q[1, 1] = 1
    Q[5, 5] = 10
    Q[6, 6] = 10
    Q_bar = np.kron(np.eye(N, dtype=int), Q)
    R = np.zeros((m, m))
    R[0, 0] = 10
    R[1, 1] = 0.001
    R_bar = np.kron(np.eye(N, dtype=int), R)
    D = np.zeros((n, l))

    sim_length = 250
    states = np.zeros((8+3, sim_length))
    controls = np.zeros((2, sim_length))

    ks = np.zeros((m*N, n*N, sim_length))
    ss = np.zeros((1, sim_length))
    dictionary = np.load("Ks_ltv_10N_9mps.npz")
    ks = dictionary['ks']
    ss = dictionary['ss']

    solver = CSSolver(n, m, l, N, u_min, u_max, mean_only=False, lti_k=True)
    solve_process = DummyProcess(target=solver.solve)
    try:
        for ii in range(int(sim_length/1)):
            t0 = time.time()
            A, B, d = ar.linearize_dynamics(xs, us)
            if B[4, 1] < 0:
                print(xs, us)
            # print(d)
            A = A.reshape((n, n, N), order='F')
            B = B.reshape((n, m, N), order='F')
            d = d.reshape((n, 1, N), order='F')
            # D = np.eye(n)
            # D[1, 1] = 0.001
            # D[2, 2] = 0.001
            D = np.tile(D.reshape((n, l, 1)), (1, 1, N))
            # A = np.eye(8)
            # A[0, 0] = 0.5
            # A[0, 4] = 0.5
            # A[4, 4] = 0.9
            # A[5, 2] = 1
            # B = np.zeros((8, 2))
            # B[2, 0] = 1
            # B[4, 1] = 1
            # B = np.vstack((np.hstack((B, np.zeros_like(B), np.zeros_like(B))), np.hstack((np.dot(A, B), B, np.zeros_like(B))), np.hstack((np.dot(A, np.dot(A, B)), np.dot(A, B), B))))
            # A = np.vstack((A, np.dot(A, A), np.dot(A, np.dot(A, A))))
            A, B, d, D = ar.form_long_matrices_LTV(A, B, d, D)
            # A, B, D = ar.form_long_matrices_LTI(A[:, :, 0], B[:, :, 0], np.zeros((8, 8)))
            # print(np.allclose(A1, A), np.allclose(B1, B))
            # nearest = np.argmin(np.abs(state[7, 0] - ss[0, :]))
            # K = ks[:, :, nearest]
            solver.populate_params(A, B, d, D, xs[:, 0], sigma_0, sigma_N_inv, Q_bar, R_bar, us[:, 0], x_target)
            # A = np.eye(8)
            # A[0, 0] = 0.5
            # A[0, 4] = 0.5
            # A[5, 2] = 1
            # B = np.zeros((8, 2))
            # B[2, 0] = 1
            # B[4, 1] = 0.01
            # solver.M.setSolverParam("numThreads", 8)
            # solver.M.setSolverParam("intpntCoTolPfeas", 1e-3)
            # solver.M.setSolverParam("intpntCoTolDfeas", 1e-3)
            # solver.M.setSolverParam("intpntCoTolRelGap", 1e-3)
            # solver.M.setSolverParam("intpntCoTolInfeas", 1e-3)
            # X = np.dot(A, x)
            # print(X.reshape((10, 8)))
            # solve_process.start()
            # solve_process.join()
            # V, K = (solver.V.level(), solver.K.level())
            try:
                V, K = solver.solve()
                K = K.reshape((m*N, n*N))
                print("K")
                print(K)
            except RuntimeError:
                print("RuntimeError")
                V = np.tile(np.array([0, -1]).reshape((-1, 1)), (N, 1)).flatten()
                K = np.zeros((m*N, n*N))
            # ks[:, :, ii] = K[:, :]
            # ss[0, ii] = state[7, 0]
            # nearest = np.argmin(np.abs(state[7, 0] - ss[0, :]))
            # K = ks[:, :, nearest]
            us = V.reshape((m, N), order='F')
            us[:, 0] = V[:m]
            t = 0
            '''
            print(xs[:, 0])
            print(us[:, 0])
            '''
            X_bar = np.dot(A, xs[:, 0]) + np.dot(B, V) + d.flatten()
            y = np.zeros((n, 1)).flatten()
            for jj in range(1):
                # print(y)
                u = V[jj*m:(jj+1)*m] + np.dot(K[jj*m:(jj+1)*m, jj*n:(jj+1)*n], y)
                u = np.where(u > u_max, u_max, u)
                u = np.where(u < u_min, u_min, u)
                states[:n, ii*1+jj] = state.flatten()
                states[n:, ii*1+jj] = cartesian.flatten()
                controls[:, ii*1+jj] = u
                # print(state)
                # print(u)
                state, cartesian = ar.update_dynamics(state, u.reshape((-1, 1)), 0.1, throttle_nn=ar.throttle, cartesian=cartesian)
                # state += np.array([0.1, 0.01, 0.01, 1, 1, 0, 0, 0]).reshape((-1, 1)) * np.random.randn(n, 1)
                y = state.flatten() - X_bar[jj * n:(jj + 1) * n]
                if jj == 0:
                    D = np.diag(y)
            # us[:, 0:1] += Ky
            # us[:, 0] = np.where(us[:, 0] > u_max, u_max, us[:, 0])
            # us[:, 0] = np.where(us[:, 0] < u_min, u_min, us[:, 0])
            print(us[:, 0])
            # state = ar.update_dynamics(state, us[:, 0:1], 0.05, throttle_nn=ar.throttle)
            xs = np.dot(A, xs[:, 0]) + np.dot(B, V) + d.flatten()
            xs = xs.reshape((n, N), order='F')
            # y = xs[:, 0:1].copy()
            xs[:, 0] = state.flatten()
            # print(xs)
            # X = np.dot(A, x) + np.dot(B, V.reshape((-1, 1)))
            # print(X.reshape((10, 8)))
            #     solver.time()
            print(time.time() - t0)
        plot(states, controls, sim_length)
        # np.savez('Ks_lti_20N_7mps.npz', ks=ks, ss=ss)
    finally:
        solver.M.dispose()


from cs_solver import CSSolver
if __name__ == '__main__':
    # u_min = np.array([-0.9, 0.1])
    # u_max = np.array([0.9, 0.9])
    # x = np.array([4., 0., 0., 50., 50., 0.1, 0., 0.]).reshape((8, 1))
    # u = np.array([0.01, 0.5]).reshape((2, 1))
    # xs = np.tile(x, (1, 10))
    # us = np.tile(u, (1, 10))
    # solver = CSSolver(8, 2, 8, 10, u_min, u_max)
    # solver2 = CSSolver(8, 2, 8, 10, u_min, u_max)
    # ar = Model(10)
    # solve_process = DummyProcess(target=solver.solve)
    # solve_process2 = DummyProcess(target=solver2.solve)
    # lin_process = DummyProcess(target=ar.linearize_dynamics, args=(xs, us))
    # solve_process.start()
    # # lin_process.start()
    # solve_process.join()
    # # lin_process.join()
    # t0 = time.time()
    # A, B, d = ar.linearize_dynamics(xs, us)
    # D = np.zeros((8, 8, 10))
    # A = A.reshape((8, 8, 10), order='F')
    # B = B.reshape((8, 2, 10), order='F')
    # d = d.reshape((8, 1, 10), order='F')
    # ar.form_long_matrices_LTV(A, B, d, D)
    # print(time.time() - t0)
    # ar = Model(1)
    # throttle = throttle_model.Net()
    # xs = np.array([4.97, -0.01, -0.069, 50, 49, 0.1, 0.0049, 0.049]).reshape((-1, 1))
    # us = np.array([-0.06377, 0.1]).reshape((-1, 1))
    # A, B, d = ar.linearize_dynamics(xs, us)
    # print(A, B)
    # # x1 = ar.update_dynamics(xs, us, 0.1, throttle_nn=ar.throttle)
    # # print(x1)
    # T = np.array([.05, 0.1, 0.15])
    # wR = np.array([49, 49, 49])
    # throttle_factor = 0.31
    # input_tensor = torch.from_numpy(np.hstack((T.reshape((-1, 1)), wR.reshape((-1, 1)) / throttle_factor))).float()
    # dwR = ar.throttle(input_tensor).detach().numpy().flatten()
    # print(dwR)
    run_simple_controller()
    # n = 8
    # m=2
    # N=10
    # x = np.array([5., 0., 0., 50., 50., 0., 0., 0.]).reshape((8, 1))
    # u = np.array([0.02, 50.]).reshape((2, 1))
    # xs = np.tile(x, (1, 1))
    # us = np.tile(u, (1, 1))
    # A, B, d = linearize_dynamics(xs, us)
    # sigma_0 = np.dot(np.dot(A, x), np.dot(A, x).T) - np.dot(x, x.T)
    # # print(sigma_0)
    # # print(A, B)
    # A, B, D = form_long_matrices_LTI(A, B, np.zeros((8,8)))
    #
    # sigma_0 = np.random.randn(n, n)
    # sigma_0 = np.dot(sigma_0, sigma_0.T)
    # sigma_0 = np.zeros((n, n))
    # # sigma_N_inv = np.linalg.inv(1e6 * sigma_0)
    # sigma_N_inv = sigma_0
    # Q = np.zeros((n, n))
    # Q[0,0] = 1000
    # Q[5,5] = 10
    # Q[6, 6] = 10
    # Q_bar = np.kron(np.eye(N,dtype=int),Q)
    # R = np.zeros((m, m))
    # R[0,0] = 10
    # R[1,1] = 0.00001
    # R_bar = np.kron(np.eye(N,dtype=int),R)
    #
    # solver = CSSolver()
    # try:
    #     solver.populate_params(A, B, D, x, sigma_0, sigma_N_inv, Q_bar, R_bar)
    #     # solver.M.setSolverParam("numThreads", 8)
    #     # solver.M.setSolverParam("intpntCoTolPfeas", 1e-3)
    #     # solver.M.setSolverParam("intpntCoTolDfeas", 1e-3)
    #     # solver.M.setSolverParam("intpntCoTolRelGap", 1e-3)
    #     # solver.M.setSolverParam("intpntCoTolInfeas", 1e-3)
    #     # X = np.dot(A, x)
    #     # print(X.reshape((10, 8)))
    #     V = solver.solve()
    #     X = np.dot(A, x) + np.dot(B, V.reshape((-1, 1)))
    #     print(X.reshape((10, 8)))
    #     #     solver.time()
    # finally:
    #     solver.M.dispose()
