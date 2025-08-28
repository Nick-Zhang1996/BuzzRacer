# code for validating strategy controller
import pickle
import numpy as np
from qpSmooth import QpSmooth
from buzzracer.tracks.rcp_track_debug import RCPTrackDebug as RCPTrack
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev, CubicSpline, interp1d

# TODO thoroughly check this


def build_lead_fun(tt_i, ss_i, vv_i, tt_j, ss_j, vv_j, dt=0.01):
    s2t_i = interp1d(ss_i, tt_i, kind='cubic')
    t2s_i = interp1d(tt_i, ss_i, kind='cubic')

    def t0_dt_2_ds_i(t0, dt): return t2s_i((t0+dt) % tt_i[-1])-t2s_i(t0 % tt_i[-1]) if t2s_i(
        (t0+dt) % tt_i[-1])-t2s_i(t0 % tt_i[-1]) > 0 else (t2s_i((t0+dt) % tt_i[-1])-t2s_i(t0 % tt_i[-1]) + ss_i[-1])

    s2t_j = interp1d(ss_j, tt_j, kind='cubic')
    t2s_j = interp1d(tt_j, ss_j, kind='cubic')
    def t0_dt_2_ds_j(t0, dt): return t2s_j((t0+dt) % tt_j[-1])-t2s_j(t0 % tt_j[-1]) if t2s_j(
        (t0+dt) % tt_j[-1])-t2s_j(t0 % tt_j[-1]) > 0 else (t2s_j((t0+dt) % tt_j[-1])-t2s_j(t0 % tt_j[-1]) + ss_j[-1])

    # Lij : max_lead(ss_i)
    L_ij = []
    for s0 in ss_i:
        max_lead = 0
        dt = 0.01
        step_size = 0.01

        t0_i = s2t_i(s0)
        t0_j = s2t_j(s0)

        # TODO: optimize this
        while (t0_dt_2_ds_i(t0_i, dt) - t0_dt_2_ds_j(t0_j, dt) > max_lead):
            max_lead = t0_dt_2_ds_i(t0_i, dt) - t0_dt_2_ds_j(t0_j, dt)
            dt += step_size
        L_ij.append(max_lead)

    plt.plot(ss_i, L_ij)
    plt.show()
    # check ss_i[-1] and ss_j[-1]
    return interp1d(ss_i, L_ij, kind='cubic'), np.min(L_ij), np.max(L_ij)


def build_statefrom_speed_profile(fulltrack, speed_profile_fun, dt=0.01):
    tt = []
    ss = []
    vv = []
    s = 0
    t = 0

    # progress in time domain
    while (s < fulltrack.raceline_len_m):
        v = fulltrack.sToV(s)
        tt.append(t)
        ss.append(s)
        vv.append(v)
        s += v*dt
        t += dt
    print(f'total t = {tt[-1]}')
    '''
    # progress in s domain
    tt = []
    ss = []
    vv = []
    s = 0
    t = 0
    ds = 0.01
    while (s<fulltrack.raceline_len_m):
        tt.append(t)
        ss.append(s)
        v = fulltrack.sToV(s)
        vv.append(v)
        t += ds/v
        s += ds
    '''
    # 2*N
    pos = np.array(splev(ss, fulltrack.raceline_s, der=0))
    d_pos = np.array(splev(ss, fulltrack.raceline_s, der=1))
    phi = np.arctan2(d_pos[1, :], d_pos[0, :])
    tt = np.array(tt)
    ss = np.array(ss)
    vv = np.array(vv)
    return (pos.T, phi.T, tt, ss, vv)


if __name__ == '__main__':
    fulltrack = QpSmooth()
    fulltrack.load()
    dt = 0.01

    # create 2 speed profile with similar laptime

    # agent i, faster in straights
    retval = fulltrack.generate_speed_profile(
        mu=0.6,
        acc_max_fun=lambda x: 5.0,
        dec_max_fun=lambda x: 5.0,
    )
    fulltrack.targetVfromU = speed_profile_fun_i = retval['speed_profile_fun']
    # this calls reconstruct Raceline, which updates sToV
    fulltrack.verify_speed_profile(speed_profile_fun=speed_profile_fun_i)
    pos_i, phi_i, tt_i, ss_i, vv_i = build_statefrom_speed_profile(
        fulltrack, speed_profile_fun_i, dt=dt)

    # agent j, faster in corners
    retval = fulltrack.generate_speed_profile(
        mu=0.9,
        acc_max_fun=lambda x: 1,
        dec_max_fun=lambda x: 1,
    )
    fulltrack.targetVfromU = speed_profile_fun_j = retval['speed_profile_fun']
    fulltrack.verify_speed_profile(speed_profile_fun=speed_profile_fun_j)
    pos_j, phi_j, tt_j, ss_j, vv_j = build_statefrom_speed_profile(
        fulltrack, speed_profile_fun_j, dt=dt)

    # calculate relative lead
    s0_to_Lij, Lij_min, Lij_max = build_lead_fun(
        tt_i, ss_i, vv_i, tt_j, ss_j, vv_j)

    # visualize L_ij

    img_track = fulltrack.draw_track()
    img_track = fulltrack.draw_raceline_with_color(
        img=img_track, thickness=10, s_to_color=lambda s: s0_to_Lij(s % ss_i[-1])/Lij_max)
    plt.imshow(img_track[:, :, ::-1])
    plt.show()

    # create mock log for visualization
    # t*car*states
    T = pos_i.shape[0]
    tt = np.linspace(0, T-1, T)*dt
    state_i = np.vstack(
        [tt, pos_i.T, phi_i, np.zeros((5, T))]).T.reshape(-1, 1, 9)
    # record two laps
    state_i = np.vstack([state_i]*2)
    T = pos_j.shape[0]
    tt = np.linspace(0, T-1, T)*dt
    state_j = np.vstack(
        [tt, pos_j.T, phi_j, np.zeros((5, T))]).T.reshape(-1, 1, 9)
    state_j = np.vstack([state_j]*2)

    T = min(state_i.shape[0], state_j.shape[0])
    log = np.hstack([state_i[:T], state_j[:T]])

    print(log.shape)
    output = open('../log/strategy/strategy_car_demo.p', 'wb')
    pickle.dump(log, output)
    output.close()

    #
