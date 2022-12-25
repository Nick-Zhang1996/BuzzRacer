# RC car dynamics
import torch
try:
    from Track import Track
except ModuleNotFoundError:
    from copg.rcvip_simulator.Track import Track
import numpy as np
import copy
from math import radians,degrees
import matplotlib.pyplot as plt

class VehicleModel():
    def __init__(self,n_batch,device,track='orca',dt=0.03):
        print('rcvip version of VehicleModel')

        self.device = device
        self.track = Track()
        if (track == 'orca'):
            self.track.loadOrcaTrack()
        elif (track == 'rcp'):
            self.track.loadRcpTrack()

        self.track_n = self.track.s.shape[0]
        self.track_s = torch.from_numpy(self.track.s).type(torch.FloatTensor).to(self.device)
        self.track_kappa = torch.from_numpy(self.track.kappa).type(torch.FloatTensor).to(self.device)
        self.track_phi = torch.from_numpy(self.track.phi).type(torch.FloatTensor).to(self.device)
        self.track_X = torch.from_numpy(self.track.X).type(torch.FloatTensor).to(self.device)
        self.track_Y = torch.from_numpy(self.track.Y).type(torch.FloatTensor).to(self.device)

        self.track_d_upper = torch.from_numpy(self.track.d_upper).type(torch.FloatTensor).to(self.device)
        self.track_d_lower = torch.from_numpy(self.track.d_lower).type(torch.FloatTensor).to(self.device)
        self.track_angle_upper = torch.from_numpy(self.track.border_angle_upper).type(torch.FloatTensor).to(self.device)
        self.track_angle_lower = torch.from_numpy(self.track.border_angle_lower).type(torch.FloatTensor).to(self.device)


        self.n_state = 6
        self.n_control = 2

        self.n_batch = n_batch
        # Model Parameters
        #self.Cm1 = 0.287
        #self.Cm2 = 0.054527
        #self.Cr0 = 0.051891
        #self.Cr2 = 0.000348

        #self.B_r = 3.3852 / 1.2
        #self.C_r = 1.2691
        #self.D_r = 1. * 0.1737 * 1.2

        #self.B_f = 2.579
        #self.C_f = 1.2
        #self.D_f = 1.05 * .192

        #self.mass = 0.041
        #self.mass = 0.041
        #self.I_z = 27.8e-6
        #self.l_f = 0.029
        #self.l_r = 0.033

        #self.L = 0.06
        #self.W = 0.03

        #self.tv_p = 0

        self.L = 0.09
        self.l_f = 0.04824
        self.l_r = self.L - self.l_f
        self.I_z = 417757e-9
        self.mass = 0.1667

        self.Ts = dt
        self.max_steering = radians(26.5)



    # advance dynamics in cartesian frame
    def dx(self, x, u):
        f = torch.empty(self.n_batch, self.n_state,device=self.device)
        # state x: X,Y,phi, v_x, v_y, omega

        phi = x[:, 2]
        v_x = x[:, 3]
        v_y = x[:, 4]
        r = x[:, 5]

        throttle = u[:,0]
        delta = u[:, 1]

        r_tar = delta * v_x / (self.l_f + self.l_r)

        [F_rx, F_ry, F_fy] = self.forceModel(x, u)

        f[:, 0] = v_x * torch.cos(phi) - v_y * torch.sin(phi)
        f[:, 1] = v_x * torch.sin(phi) + v_y * torch.cos(phi)
        f[:, 2] = r
        # orca
        #f[:, 3] = 1 / self.mass * (F_rx - F_fy * torch.sin(delta) + self.mass * v_y * r)
        #f[:, 5] = 1 / self.I_z * (F_fy * self.l_f * torch.cos(delta) - F_ry * self.l_r + self.tv_p * (r_tar - r))
        # rcvip
        f[:,3] =  F_rx / self.mass
        f[:, 4] = 1 / self.mass * (F_ry + F_fy * torch.cos(delta) - self.mass * v_x * r)
        f[:, 5] = 1 / self.I_z * (F_fy * self.l_f * torch.cos(delta) - F_ry * self.l_r)
        return f

    def forceModel(self, x, u):
        # cartesian: X,Y,phi, vx,vy,w
        # curvilinear: progress, lateral_err, heading_err, vx,vy,w
        v_x = x[:, 3]
        v_y = x[:, 4]
        r = x[:, 5]

        # throttle, steering
        D = u[:, 0]
        delta = u[:, 1]

        alpha_f = -torch.atan((self.l_f * r + v_y) / (v_x+1e-5)) + delta
        alpha_r = torch.atan((self.l_r * r - v_y) / (v_x+1e-5))

        # orca
        #F_rx = self.Cm1 * D - self.Cm2*v_x*D - self.Cr2*v_x**2 - self.Cr0
        #F_ry = self.D_r * torch.sin(self.C_r * torch.atan(self.B_r * alpha_r))
        #F_fy = self.D_f * torch.sin(self.C_f * torch.atan(self.B_f * alpha_f))

        # rcvip
        #F_rx = 6.17*(D - v_x/15.2 -0.333) * self.mass
        F_rx = D*(3.0-v_x) * self.mass
        F_fy = self.tireCurve(alpha_f) * self.mass * 9.8 *self.l_r/(self.l_r+self.l_f)
        F_ry = 1.15*self.tireCurve(alpha_r) * self.mass * 9.8 *self.l_f/(self.l_r+self.l_f)

        return F_rx, F_ry, F_fy

    def tireCurve(self,alpha):
        C = 1.6
        B = 2.3
        D = 1.1
        retval = D * torch.sin( C * torch.atan(B *alpha)) 
        return retval

    # refine discretization
    def dynModel(self, x, u):

        k1 = self.dx(x, u)
        k2 = self.dx(x + self.Ts / 2. * k1, u)
        k3 = self.dx(x + self.Ts / 2. * k2, u)
        k4 = self.dx(x + self.Ts * k3, u)

        x_next = x + self.Ts * (k1 / 6. + k2 / 3. + k3 / 3. + k4 / 6.)

        return x_next

    def kinModel(self,x,u):
        k1 = self.dxkin(x, u)
        x_next = x + self.Ts * k1
        return x_next

    def kinModelCurve(self,x,u):

        k1 = self.dxkin(x, u)
        k2 = self.dxkin(x + self.Ts / 2. * k1, u)
        k3 = self.dxkin(x + self.Ts / 2. * k2, u)
        k4 = self.dxkin(x + self.Ts * k3, u)

        x_next = x + self.Ts * (k1 / 6. + k2 / 3. + k3 / 3. + k4 / 6.)

        return x_next

    def dynModel(self, x, u):

        k1 = self.dxCurve(x, u)

        x_next = x + self.Ts * k1

        return x_next

    def dynModelCurve(self, x, u):

        k1 = self.dxCurve(x, u)
        k2 = self.dxCurve(x + self.Ts / 2. * k1, u)
        k3 = self.dxCurve(x + self.Ts / 2. * k2, u)
        k4 = self.dxCurve(x + self.Ts * k3, u)

        x_next = x + self.Ts * (k1 / 6. + k2 / 3. + k3 / 3. + k4 / 6.)

        return x_next

    # NOTE used
    # x: curvlinear states : (progress, lateral_err, orientation_err, vx,vy
    def dynModelBlendBatch(self, x, u_unclipped):
        '''
        blend_ratio = (x[:,3] - 0.3)/(0.2)
        # lambda_blend = np.min([np.max([blend_ratio,0]),1])
        blend_max = torch.max(torch.cat([blend_ratio.view(-1,1), torch.zeros(blend_ratio.size(0),1)],dim=1),dim=1)
        blend_min = torch.min(torch.cat([blend_max.values.view(-1, 1), torch.ones(blend_max.values.size(0), 1)], dim=1), dim=1)
        lambda_blend = blend_min.values
        '''

        #t.s('dyn prep')
        if not torch.is_tensor(x):
            x = torch.tensor(x).reshape(-1,6)
            u_unclipped = torch.tensor(u_unclipped).reshape(-1,2)
        blend_ratio = (x[:,3]>0.05).float()
        lambda_blend = blend_ratio

        u = u_unclipped
        u[:,0] = torch.clamp(u_unclipped[:,0],-1,1) # throttle
        # original curoff is [-1,1]
        u[:,1] = torch.clamp(u_unclipped[:,1],-self.max_steering,self.max_steering) # steering angle

        # Kinematic Model
        v_x = x[:,3]
        v_y = x[:, 4]
        x_kin = torch.cat([x[:,0:3], torch.sqrt(v_x*v_x + v_y*v_y).reshape(-1,1)],dim =1)
        #t.e('dyn prep')

        #t.s('kin model')
        #x_kin_state = self.kinModelCurve(x_kin,u)
        x_kin_state = self.kinModel(x_kin,u)

        delta = u[:, 1]
        beta = torch.atan(self.l_r * torch.tan(delta) / (self.l_f + self.l_r))
        v_x_state = x_kin_state[:,3] * torch.cos(beta) # V*cos(beta)
        v_y_state = x_kin_state[:,3] * torch.sin(beta) # V*sin(beta)
        yawrate_state = v_x_state * torch.tan(delta)/(self.l_f + self.l_r)

        x_kin_full = torch.cat([x_kin_state[:,0:3],v_x_state.view(-1,1),v_y_state.view(-1,1), yawrate_state.view(-1,1)],dim =1)
        #t.e('kin model')

        # Dynamic Model
        #t.s('dyn model')
        #x_dyn = self.dynModelCurve(x,u)
        x_dyn = self.dynModel(x,u)
        #t.e('dyn model')

        return  (x_dyn.transpose(0,1)*lambda_blend + x_kin_full.transpose(0,1)*(1-lambda_blend)).transpose(0,1)

    def dxkin(self, x, u):

        fkin = torch.empty(x.size(0), 4,device=self.device)

        s = x[:,0] #progress
        d = x[:,1] #horizontal displacement
        mu = x[:, 2] #orientation
        v_x = v = x[:, 3]

        throttle = u[:,0]
        delta = u[:, 1]

        kappa = self.getCurvature(s)

        beta = torch.atan(self.l_r*torch.tan(delta)/(self.l_f + self.l_r))

        fkin[:, 0] = (v*torch.cos(beta + mu))/(1.0 - kappa*d)   # s_dot
        fkin[:, 1] = v*torch.sin(beta + mu) # d_dot
        fkin[:, 2] = v*torch.sin(beta)/self.l_r - kappa*(v*torch.cos(beta + mu))/(1.0 - kappa*d)
        '''
        slow_ind =  v<=0.1
        D_0 = (self.Cr0 + self.Cr2*v*v)/(self.Cm1 - self.Cm2 * v)
        D_slow  = torch.max(D_0,u[:,0])
        D_fast = u[:,0]
        D = D_slow*slow_ind + D_fast*(~slow_ind)
        '''

        #F_rx = 6.17*(throttle - v_x/15.2 -0.333) * self.mass
        F_rx = throttle*(3.0-v_x) * self.mass
        fkin[:, 3] = 1 / self.mass * F_rx

        return fkin

    def dxCurve_blend(self, x, u):

        f = torch.empty(self.n_batch, self.n_state,device=self.device)

        s = x[:,0] #progress
        d = x[:,1] #horizontal displacement
        mu = x[:, 2] #orientation
        v_x = x[:, 3]
        v_y = x[:, 4]
        r = x[:, 5] #yawrate

        delta = u[:, 1]

        r_tar = delta * v_x / (self.l_f + self.l_r)

        blend_ratio = (v_x - 0.3)/(0.2)

        lambda_blend = np.min([np.max([blend_ratio,0]),1])
        kappa = self.getCurvature(s)

        if lambda_blend<1:
            fkin = torch.empty(self.n_batch, self.n_state,device=self.device)

            v = np.sqrt(v_x*v_x + v_y*v_y)
            beta = torch.tan(self.l_r*torch.atan(delta/(self.l_f + self.lr)))

            fkin[:, 0] = (v_x * torch.cos(mu) - v_y * torch.sin(mu))/(1.0 - kappa*d)   # s_dot
            fkin[:, 1] = v_x * torch.sin(mu) + v_y * torch.cos(mu) # d_dot
            fkin[:, 2] = v*torch.sin(beta)/self.l_r - kappa*((v_x * torch.cos(mu) - v_y * torch.sin(mu))/(1.0 - kappa*d))
            v_dot = 1 / self.mass * (self.Cm1 * u[:, 0] - self.Cm2 * v_x * u[:, 0])

            fkin[:, 3] = 1 / self.mass * (self.Cm1 * u[:, 0] - self.Cm2 * v_x * u[:, 0])
            fkin[:, 4] = delta * fkin[:, 3] * self.l_r / (self.l_r + self.l_f)
            fkin[:, 5] = delta * fkin[:, 3] / (self.l_r + self.l_f)
            if lambda_blend ==0:
                return fkin

        if lambda_blend>0:
            [F_rx, F_ry, F_fy] = self.forceModel(x, u)

            f[:, 0] = (v_x * torch.cos(mu) - v_y * torch.sin(mu))/(1.0 - kappa*d)
            f[:, 1] =  v_x * torch.sin(mu) + v_y * torch.cos(mu)
            f[:, 2] = r - kappa*((v_x * torch.cos(mu) - v_y * torch.sin(mu))/(1.0 - kappa*d))
            f[:, 3] = 1 / self.mass * (F_rx - F_fy * torch.sin(delta) + self.mass * v_y * r)
            f[:, 4] = 1 / self.mass * (F_ry + F_fy * torch.cos(delta) - self.mass * v_x * r)
            f[:, 5] = 1 / self.I_z * (F_fy * self.l_f * torch.cos(delta) - F_ry * self.l_r + self.tv_p * (r_tar - r))
            if lambda_blend ==1:
                return f

        return f*lambda_blend + (1-lambda_blend)*fkin


    # advance dynamics in curvilinear frame
    def dxCurve(self, x, u):
        f = torch.empty(x.size(0), self.n_state,device=self.device)

        s = x[:,0] #progress
        d = x[:,1] #horizontal displacement
        mu = x[:, 2] #orientation
        v_x = x[:, 3]
        v_y = x[:, 4]
        r = x[:, 5] #yawrate

        delta = u[:, 1]

        r_tar = delta * v_x / (self.l_f + self.l_r)

        [F_rx, F_ry, F_fy] = self.forceModel(x, u)

        kappa = self.getCurvature(s)

        f[:, 0] = (v_x * torch.cos(mu) - v_y * torch.sin(mu))/(1.0 - kappa*d)
        f[:, 1] =  v_x * torch.sin(mu) + v_y * torch.cos(mu)
        f[:, 2] = r - kappa*((v_x * torch.cos(mu) - v_y * torch.sin(mu))/(1.0 - kappa*d))

        # orca
        #f[:, 3] = 1 / self.mass * (F_rx - F_fy * torch.sin(delta) + self.mass * v_y * r)
        #f[:, 4] = 1 / self.mass * (F_ry + F_fy * torch.cos(delta) - self.mass * v_x * r)
        #f[:, 5] = 1 / self.I_z * (F_fy * self.l_f * torch.cos(delta) - F_ry * self.l_r + self.tv_p * (r_tar - r))

        # rcvip
        f[:, 3] = F_rx / self.mass
        f[:, 4] = 1 / self.mass * (F_ry + F_fy * torch.cos(delta) - self.mass * v_x * r)
        f[:, 5] = 1 / self.I_z * (F_fy * self.l_f * torch.cos(delta) - F_ry * self.l_r)
        return f

    def fromStoIndexBatch(self,s_in):

        s = s_in

        i_nan = (s != s)
        i_nan += (s >= 1e10) + (s <= -1e10)
        if torch.sum(i_nan) > 0:
            for i in range(self.n_batch):
                if i_nan[i]:
                    s[i] = 0
        # s[i_nan] = torch.zeros(torch.sum(i_nan))
        k = 0
        if torch.max(s) > self.track_s[-1] or torch.min(s) < 0:
            s = torch.fmod(s,self.track_s[-1])
            # i_wrapdown = (s > self.track_s[-1]).type(torch.FloatTensor)
            i_wrapup = (s < 0).float()

            s = s + i_wrapup * self.track_s[-1]

            if torch.max(s) > self.track_s[-1] or torch.min(s) < 0:
                s = torch.max(s, torch.zeros(self.n_batch))
                s = torch.min(s, self.track_s[-1] * torch.ones(self.n_batch))

            # print(s-s_in)

        index = (torch.floor(s / self.track.diff_s)).to(torch.long)
        if torch.min(index) < 0:
            print(index)

        rela_proj = (s - self.track_s[index]) / self.track.diff_s

        next_index = index + 1
        i_index_wrap = (next_index < self.track.N).to(torch.long)
        next_index = torch.fmod(next_index,self.track.N)# * i_index_wrap

        return index, next_index, rela_proj

    def getCurvature(self, s):
        index, next_index, rela_proj = self.fromStoIndexBatch(s)

        kappa = self.track_kappa[index] + rela_proj * (self.track_kappa[next_index] - self.track_kappa[index])

        return kappa

    def getTrackHeading(self,s):
        index, next_index, rela_proj = self.fromStoIndexBatch(s)
        phi = self.track_phi[index] + rela_proj * (self.track_phi[next_index] - self.track_phi[index])
        return phi

    def getLocalBounds(self,s):
        index, next_index, rela_proj = self.fromStoIndexBatch(s)
        d_upper = self.track_d_upper[index] + \
                  rela_proj * (self.track_d_upper[next_index] - self.track_d_upper[index])
        d_lower = self.track_d_lower[index] +\
                  rela_proj * (self.track_d_lower[next_index] - self.track_d_lower[index])

        angle_upper = self.track_angle_upper[index] + \
                      rela_proj * (self.track_angle_upper[next_index] - self.track_angle_upper[index])
        angle_lower = self.track_angle_lower[index] + \
                      rela_proj * (self.track_angle_lower[next_index] - self.track_angle_lower[index])

        return d_upper, d_lower,angle_upper,angle_lower


    def fromGlobalToLocal(self,state_global):
        track_ref_pos = np.vstack([self.track_X, self.track_Y]).T
        track_s = self.track_s.numpy()
        ds = track_s[1]-track_s[0]
        # first index and last are same
        track_n = self.track_n - 1

        state_global = np.array(state_global).flatten()
        assert(state_global.shape == (6,))
        x = state_global[0]
        y = state_global[1]
        car_pos = np.hstack([x,y])

        abs_heading = state_global[2]
        v_x = state_global[3]
        v_y = state_global[4]
        omega = state_global[5]

        # find closest index
        dist = ((x-track_ref_pos[:,0])**2 + (y-track_ref_pos[:,1])**2)**0.5
        index = np.argmin(dist)
        track_tangent = track_ref_pos[(index+1)%track_n] - track_ref_pos[(index-1)%track_n]
        track_tangent = track_tangent/np.linalg.norm(track_tangent)
        # vector track_ref -> car
        ref = track_ref_pos[index]
        r_ref_car = car_pos - ref

        mid_ref = np.dot(track_tangent, r_ref_car) * track_tangent + ref
        progress = self.track_s[index] + np.dot(track_tangent, r_ref_car)
        progress = (progress + track_s[-1]) % track_s[-1]
        lateral = np.cross(track_tangent, r_ref_car)
        # (-1,1)
        ratio = np.dot(track_tangent, r_ref_car) / ds
        
        forward_diff  =  self.wrap(self.track_phi[(index+1)%track_n] - self.track_phi[index])
        backward_diff =  self.wrap(self.track_phi[index] - self.track_phi[(index-1)%track_n])
        ref_heading = self.track_phi[index] + (ratio > 0) * ratio * forward_diff + (ratio < 0) * ratio*backward_diff

        rel_heading = (abs_heading - ref_heading + np.pi) % (2*np.pi) - np.pi
        retval = np.hstack([progress,lateral,rel_heading, v_x,v_y,omega])
        return retval

    def fromLocalToGlobal(self, state_local):
        track_ref_pos = np.vstack([self.track_X, self.track_Y]).T
        track_s = self.track_s.numpy()
        ds = track_s[1]-track_s[0]
        # first index and last are same
        track_n = self.track_n - 1

        state_local = np.array(state_local).reshape(6)
        s = (state_local[0] + track_s[-1]) % track_s[-1]
        d = state_local[1]
        rel_heading = state_local[2]
        v_x = state_local[3]
        v_y = state_local[4]
        omega = state_local[5]
        index = np.searchsorted(track_s,s) - 1
        
        ratio = (s-track_s[index])/ds
        assert (ratio>=0)
        assert (ratio<=1)
        track_tangent = track_ref_pos[(index+1)%track_n] - track_ref_pos[(index-1)%track_n]
        track_tangent = track_tangent/np.linalg.norm(track_tangent)
        mid_ref_pos = track_ref_pos[index] + track_tangent*ds*ratio
        d_phi = (self.track_phi[(index+1)%track_n] - self.track_phi[index] + np.pi ) % (2*np.pi) -np.pi
        ref_heading = self.track_phi[index] + ratio* d_phi
        normal_dir = np.array((-np.sin(ref_heading), np.cos(ref_heading)))
        car_pos = mid_ref_pos + d* normal_dir
        abs_heading = (ref_heading + rel_heading + np.pi)%(2*np.pi) - np.pi
        state_global = [car_pos[0], car_pos[1], abs_heading, v_x, v_y, omega]
        return np.array(state_global)


    # state_local: s,d,mu, v_x,v_y,r (progress, lateral_err, rel_heading, vx,vy,omega)
    # return: state_global (x,y,abs_heading,vx,vy,omega)
    # TODO redo this function
    def fromLocalToGlobalOld(self,state_local):
        state_local = torch.from_numpy(state_local.reshape(1,-1))
        s = state_local[:,0]
        d = state_local[:,1]
        mu = state_local[:,2]
        v_x = state_local[:, 3]
        v_y = state_local[:, 4]
        r = state_local[:, 5]
        index, next_index, rela_proj = self.fromStoIndexBatch(s)
        vec_track = torch.empty(self.n_batch,2,device=self.device)
        vec_track[:, 0] = (self.track_X[next_index] - self.track_X[index])* rela_proj
        vec_track[:, 1] = (self.track_Y[next_index] - self.track_Y[index])* rela_proj

        pos_index = torch.empty(self.n_batch,2,device=self.device)
        pos_index[:, 0] = self.track_X[index]
        pos_index[:, 1] = self.track_Y[index]

        pos_center = pos_index + vec_track

        phi_0 = self.track_phi[index]
        # phi_1 = self.track_phi[next_index]
        #phi = phi_0
        phi = self.getTrackHeading(s)
        #self.track_phi[index] + rela_proj * (self.track_phi[next_index] - self.track_phi[index])

        pos_global = torch.empty(self.n_batch,2,device=self.device)
        pos_global[:, 0] = pos_center[:, 0] - d * torch.sin(phi)
        pos_global[:, 1] = pos_center[:, 1] + d * torch.cos(phi)

        heading = phi + mu

        # heading = torch.fmod(heading,2*np.pi)

        phi_ref = self.track_phi[index]
        upwrap_index = ((phi_ref - heading)>1.5*np.pi).type(torch.FloatTensor)
        downwrap_index = ((phi_ref - heading)<-1.5*np.pi).type(torch.FloatTensor)
        heading = heading - 2*np.pi*downwrap_index + 2*np.pi*upwrap_index

        upwrap_index = ((phi_ref - heading) > 1.5 * np.pi).type(torch.FloatTensor)
        downwrap_index = ((phi_ref - heading) < -1.5 * np.pi).type(torch.FloatTensor)
        heading = heading - 2 * np.pi * downwrap_index + 2 * np.pi * upwrap_index

        x_global = torch.empty(self.n_batch,self.n_state,device=self.device)
        x_global[:, 0] = pos_global[:, 0]
        x_global[:, 1] = pos_global[:, 1]
        x_global[:, 2] = heading
        x_global[:, 3] = v_x
        x_global[:, 4] = v_y
        x_global[:, 5] = r

        return x_global

    def wrap(self,angle):
        return (angle + np.pi)%(2*np.pi)-np.pi


from track.TrackFactory import TrackFactory
if __name__=='__main__':
    vehicle_model = VehicleModel(1,'cpu','rcp')
    # single test
    local_state = np.array([11.40600837,-0.07746057, 0.36203362,0,0,0])
    global_state = vehicle_model.fromLocalToGlobal(local_state).flatten()
    print('')

    new_local_state = vehicle_model.fromGlobalToLocal(global_state)
    print('')
    new_global_state = vehicle_model.fromLocalToGlobal(new_local_state).flatten()
    print('')


    # test position error
    pos_err = np.linalg.norm(new_global_state[:2] - global_state[:2])
    heading_err = np.linalg.norm(new_global_state[2] - global_state[2])
    track_len = vehicle_model.track_s[-1]
    progress_err = (new_local_state[0] - local_state[0] + track_len/2) % track_len - track_len/2
    print(f'pos_err {pos_err}')
    print(f'heading {heading_err}')
    print(f'progress err {progress_err}')

    print(f'local_state: {local_state}')
    print(f'local_state: {new_local_state}')
    print(f'global_state: {global_state}')
    print(f'global_state: {new_global_state}')

    print(f'diff local_state: {new_local_state-local_state}')
    print(f'diff global_state: {new_global_state-global_state}')

    # batch test
    pos_err_vec = []
    heading_err_vec = []
    track_len_vec = []
    progress_err_vec = []
    for i in range(10000):
        s = np.random.uniform(0,vehicle_model.track_s[-1])
        d = np.random.uniform(-0.1,0.1)
        mu = np.random.uniform(-radians(60),radians(60))
        local_state = np.array((s,d,mu,0,0,0))
        global_state = vehicle_model.fromLocalToGlobal(local_state).flatten()
        new_local_state = vehicle_model.fromGlobalToLocal(global_state)
        new_global_state = vehicle_model.fromLocalToGlobal(new_local_state).flatten()


        # test position error
        pos_err = np.linalg.norm(new_global_state[:2] - global_state[:2])
        heading_err = np.linalg.norm(new_global_state[2] - global_state[2])
        track_len = vehicle_model.track_s[-1]
        progress_err = (new_local_state[0] - local_state[0] + track_len/2) % track_len - track_len/2

        if (pos_err > 1e-2 or heading_err>1e-2):
            print(f'local_state: {local_state}')
            print(f'local_state: {new_local_state}')
            print(f'diff local_state: {new_local_state-local_state}')
            print('\n')

            print(f'global_state: {global_state}')
            print(f'global_state: {new_global_state}')

            print(f'diff global_state: {new_global_state-global_state}')
            breakpoint()

        pos_err_vec.append(pos_err)
        heading_err_vec.append(heading_err)
        track_len_vec.append(track_len)
        progress_err_vec.append(progress_err)

    print(f'pos_err {np.max(pos_err_vec)}')
    print(f'heading {np.max(heading_err_vec)}')
    print(f'progress err {np.max(progress_err_vec)}')


