# Base class for residual game
# for an example of a subclass, see UnstructuredDriving.py

import os
import numpy as np
from time import time
from abc import ABC,abstractmethod
from scipy import interpolate
import scipy.sparse # sparse matrix operations

from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from itertools import chain

from common import *
from util.TimeUtil import TimeUtil
#from src.build.particle_game import ParticleGame
# FIXME: Definitely need to do, TODO: will probably do, NOTE: maybe?
# TODO for cpp, change gradient for barrier function to cap at 1e20 instead of 1e10
# TODO we should maybe build a map for h(xi, xj) values

class ResidualGame(PrintObject,ABC):
    DEBUG = False
    USE_CPP = True
    FORCE_PYTHON_SOLVER = False
    CPP_DEBUG = False

    @abstractmethod
    def __init__(self):
        ''' example of a constructor '''
        # application specific parameters, to be overridden in subclass
        # the numbers here are arbitrary

        # number of agents
        self.N = 0
        # horizon length, excluding x0
        self.T = 0
        # discretization time step length
        self.dt = dt = 0.1

        # dimension of x(state) and u(control) for single agent
        self.n = 0
        self.m = 0
        # initial state, dim: N*n
        self.x0 = np.zeros((self.N,self.n))

        # initialize default parameters
        self.init()

        # max iterations
        self.iterations = 50
        self.guess = np.zeros((self.T,self.N,self.m))


    def init(self):
        ''' setup some dynamic solver parameters that changes between iterations, call this funtion to reset the solver '''
        # solver tuning parameters
        # barrier function scaling schedule
        self.rho = 10.0 * 2
        self.rho_b = 1.0 # 2.0
        # backtracking line search param
        self.bc_a = 0.1 #alpha
        self.bc_b = 0.5 #beta
        self.backtracking_max_iter = 20

        # NOTE this is not implemented in cpp
        self.dynamics_residual_weight = 1.0

        # solver variables
        self.frame_vec = []
        self.profiler = TimeUtil(False)
        #self.print_debug_enable()
        self.tolerance = 5e-4
        self.residual_vec = []

    def setup(self):
        # subclass responsible for loading specific cpp/eigen module
        # and setting x0
        self.print_info(" ---------------------------------------------------------------------------- ")
        self.print_info(" subclass did not define custom setup function, cpp module likely unavailable ")
        self.print_info(" ---------------------------------------------------------------------------- ")
        # example usage:
        '''
        if (self.USE_CPP):
            self.cpp = ParticleGame(...)
            self.cpp.set_x0(self.x0)
        '''

    def solve(self,save_gif=False,visualize=False,animate=False):
        ''' main entry point for solver, will call cpp version if available, will fallback to python if cpp does not provide a solution,
            I forgot why I did the fallback
        '''
        #TODO does cpp lscg fallback to cpp sparseQR?


        self.print_ok(f'USE_CPP: {self.USE_CPP}')
        self.print_ok(f'FORCE_PYTHON_SOLVER: {self.FORCE_PYTHON_SOLVER}')

        N = self.N; T = self.T; n = self.n; m = self.m
        # y: x(T*N*n) ,u(T*N*m), lambda(T,N,n),mu(T,N,N)
        self.print_debug(f'primal variables:{(T*N*n) +(T*N*m)} dual variables:{(N*T*n)+(T*N*N)}')

        u_ref = self.guess
        # x_ref = x_1 .. x_T, NOTE the array index is offset from the math notation
        x_ref = self.rollout(self.x0,u_ref)
        lambda_ref = np.zeros((T,N,self.n))
        # defined for all h_k_i_j, but all values may not be used
        mu_ref = np.zeros((T,N,N))
        self.visualize(u_ref,visualize=visualize, animate=animate,gif_prefix='before')
        t0 = time()
        t = self.profiler
        has_converged = False
        for i in range(self.iterations):
            self.print_info(f'------ iter {i+1} ------')
            t.s()
            if (self.USE_CPP and not self.FORCE_PYTHON_SOLVER):
                t.s('cpp step')
                try:
                    try:
                        retval = self.cpp.step(x_ref, u_ref, lambda_ref, mu_ref)
                        x_ref, u_ref, lambda_ref, mu_ref = [np.array(val) for val in retval]
                        '''
                        # FIXME debug
                        h_plus_mask = self.getHplusMask(x_ref)
                        r0 = self.r(x_ref,u_ref,lambda_ref,mu_ref,h_plus_mask)
                        r0_norm = np.linalg.norm(r0)
                        self.print_debug(f' residual = {r0_norm}')
                        '''
                        # put update here because in case solver failed, self.step() will call cpp.post_step_update()
                        self.cpp.post_step_update()
                    except RuntimeError as e:
                        self.print_warning('-----------------------------------')
                        self.print_warning(f'cpp.step() error: {e}')
                        self.print_warning('-----------------------------------')
                        x_ref, u_ref, lambda_ref, mu_ref = self.step(x_ref,u_ref,lambda_ref,mu_ref)
                except StopIteration as e:
                    self.print_ok(e)
                    if 'criteria met' in str(e):
                        has_converged = True
                    break
                finally:
                    t.e('cpp step')
            else:
                try:
                    x_ref, u_ref, lambda_ref, mu_ref = self.step(x_ref,u_ref,lambda_ref,mu_ref)
                except StopIteration as e:
                    self.print_ok(e)
                    has_converged = True
                    break
            '''
            if (np.any(mu_ref<-1e-8)):
                h_plus_mask = self.getHplusMask(x_ref)
                ratio = np.sum(mu_ref<-1e-8) / mu_ref.flatten().shape[0]
                ratio_plus = np.sum(mu_ref[h_plus_mask]<-1e-8) / mu_ref.flatten().shape[0]
                self.print_debug(f'negative lambda {ratio, ratio_plus}')
            '''
            # NOTE may not be necessary
            x_ref = self.rollout(self.x0,u_ref)
            t.e()
            #self.print_info(f'------ {N} agents, iter {i} ------')

        t_solve = time()-t0
        self.print_info(f'Total solve time: {t_solve}s')
        if (i == self.iterations-1):
            self.print_warning(f' algorithm did not reach stopping criterion ')
        full_x_ref = np.vstack([self.x0[np.newaxis,:,:],x_ref])
        self.visualize(u_ref,full_x_ref,visualize,save_gif,animate,gif_prefix='after')

        return u_ref, full_x_ref, has_converged

    def step(self,x_ref,u_ref,lambda_ref,mu_ref):
        t = self.profiler
        t.s('setup')
        N = self.N; T = self.T; n = self.n; m = self.m
        dim_x = T*N*n; dim_u = T*N*m
        # r0 + Dr*dr = 0
        h_plus_mask = self.getHplusMask(x_ref)

        r0 = self.r(x_ref,u_ref,lambda_ref,mu_ref,h_plus_mask)
        y0 = np.hstack([x_ref.flatten(), u_ref.flatten(), lambda_ref.flatten(), mu_ref.flatten()])
        # x,u,lamda,mu = split_y(y)
        split_y = lambda y: (y[:T*N*n].reshape(T,N,n), y[T*N*n:T*N*n + T*N*m].reshape(T,N,m), y[T*N*n + T*N*m:T*N*n + T*N*m + N*T*n].reshape(T,N,n), y[T*N*n + T*N*m + N*T*n:].reshape(T,N,N))
        r_y_fun = lambda y: self.r(*split_y(y),h_plus_mask)

        t.e('setup')
        Dr = self.dr_dy( x_ref, u_ref, lambda_ref, mu_ref, h_plus_mask)
        # after we remove the cols associated with unused mu, Dr will be square

        if (self.DEBUG):
            t0 = time()
            Dr_alt = jacobianNumerical(r_y_fun,y0,dim=r0.shape[0])
            self.print_debug(f't: Dr numerical {time()-t0}')
            self.print_debug(np.linalg.norm(Dr-Dr_alt))
            assert(np.linalg.norm(Dr-Dr_alt)<1e-4)

        # find newton direction, dense matrix
        '''
        t.s('lstsq')
        dy, residuals, rank, s = np.linalg.lstsq(Dr,-r0)
        t.e('lstsq')
        '''
        # find newton direction, Sparse lsqr
        '''
        t.s('sparse-lstsq')
        sparse_Dr = scipy.sparse.csc_matrix(Dr, dtype=float)
        dy, istop, itn, normr = scipy.sparse.linalg.lsqr(sparse_Dr,-r0)[:4]
        t.e('sparse-lstsq')
        '''

        # remove zero col/rows first, then use Sparse lsqr
        t.s('nonzero reduction')
        nonzero_rows = np.nonzero(np.sum(np.abs(Dr),axis=1))[0]
        nonzero_cols = np.nonzero(np.sum(np.abs(Dr),axis=0))[0]
        reduced_Dr = Dr[nonzero_rows,:][:,nonzero_cols]
        t.e('nonzero reduction')
        '''
        # NOTE debug heatmap
        abs_matrix = np.abs(reduced_Dr)
        # Plotting the heatmap
        plt.imshow(abs_matrix, cmap='viridis', interpolation='none')
        # Adding a color bar
        plt.colorbar(label='Absolute Value')
        plt.title('Heatmap of Matrix Values')
        plt.xlabel('Column Index')
        plt.ylabel('Row Index')
        plt.show()
        '''

        # use python's Sparse lsqr
        t.s('reduced-sparse-lstsq')
        sparse_Dr = scipy.sparse.csc_matrix(reduced_Dr, dtype=float)
        reduced_dy, istop, itn, normr = scipy.sparse.linalg.lsqr(sparse_Dr,-r0[nonzero_rows])[:4]
        t.e('reduced-sparse-lstsq')
        self.print_debug(f'Solver status: {"exact solution" if istop==1 else "Least Square Solution"}, iterations: {itn}')
        # use cpp's sparse QR
        '''
        t.s('cpp SparseQR')
        reduced_dy_sqr = self.cpp.SparseQR(reduced_Dr, -r0[nonzero_rows])
        t.e('cpp SparseQR')
        '''

        # use cpp's lscg (fastest)
        '''
        if (self.USE_CPP):
            t.s('cpp lscg')
            reduced_dy = reduced_dy_lscg = self.cpp.LeastSquaresConjugateGradient(reduced_Dr, -r0[nonzero_rows])
            t.e('cpp lscg')
        '''

        dy = np.zeros_like(y0)
        dy[nonzero_cols] = reduced_dy.flatten()

        '''
        # FIXME DEBUG - statistics on nonzero entries
        self.print_debug(f'nonzero rows: {len(nonzero_rows)}, ratio {len(nonzero_rows)/Dr.shape[0]}')
        self.print_debug(f'nonzero cols: {len(nonzero_cols)}, ratio {len(nonzero_cols)/Dr.shape[1]}')
        total_entries = Dr.shape[0]*Dr.shape[1]
        nonzero_entries = len(np.nonzero(Dr.flatten())[0])
        self.print_debug(f' nonzero entries:  {nonzero_entries/total_entries}')
        '''

        # projection onto dynamics null space
        # extract control constraint F
        # assert that x,u are separated from the rest
        # F @ [x,u] = Fx @ x + Fu @ u= -r_F
        # NOTE this is extremely expensive, only do this if we can't obtain an exact solution 
        if (istop == 2):
            index = 0
            for i in range(self.N):
                index += T*n + T*m
                # x: T*N*n
                x_indices = list(chain.from_iterable([list(range(t*N*n+i*n,t*N*n+(i+1)*n)) for t in range(T)]))
                # u: T*N*m
                u_indices = list(chain.from_iterable([list(range(dim_x+t*N*m+i*m,dim_x+t*N*m+(i+1)*m)) for t in range(T)]))

                Fx = Dr[index:index+n*T,x_indices]
                Fu = Dr[index:index+n*T,u_indices]
                dx_i = dy[x_indices].flatten()
                du_i = dy[u_indices].flatten()
                F = np.hstack([Fx, Fu])
                z = np.hstack([dx_i,du_i])[:,np.newaxis]
                FFT_inv = np.linalg.inv( F @ F.T) # TODO add regularization if this in singular, or use pseudoinverse
                z_null = (np.eye(z.shape[0]) - F.T @ FFT_inv @ F) @ z
                dx_i_after = z_null[:dx_i.shape[0],0]
                du_i_after = z_null[dx_i.shape[0]:,0]

                apriori = Fu @ du_i + Fx @ dx_i
                posterior = Fu @ du_i_after + Fx @ dx_i_after
                #self.print_debug(f'dynamics correction residuals {np.linalg.norm(apriori)} -> {np.linalg.norm(posterior)}')
                #self.print_debug(f'du change {np.linalg.norm(dy[u_indices]-du_i_after)}')
                #self.print_debug(f'dx change {np.linalg.norm(dy[x_indices]-dx_i_after)}')
                dy[u_indices] = du_i_after
                dy[x_indices] = dx_i_after
                index += n*T + np.sum(h_plus_mask[:,i])

        # Backtracking line search
        t.s('line search')
        apriori_h_res = self.getCollisionResidual(x_ref) # NOTE optimize?
        # backtracking line search
        step = 1.0 # step size
        dy = dy.flatten()
        r0_norm = np.linalg.norm(r0)
        flag_no_step = True
        for i in range(self.backtracking_max_iter):
            y_new = y0+step*dy
            x_new,u_new,_,_ = split_y(y_new)
            # NOTE do we still need to rollout here? maybe for nonlinear dynamics?
            #x_new = self.rollout(self.x0, u_new)
            #y_new[:dim_x] = x_new.flatten()
            search_h_res = self.getCollisionResidual(x_new)
            r_t = r_y_fun(y_new)
            r_t_norm = np.linalg.norm(r_t)
            if (r_t_norm > (1-self.bc_a*step)*r0_norm or search_h_res > apriori_h_res):
                step *= self.bc_b
            else:
                flag_no_step = False
                break
        t.e('line search')

        '''
        # debug
        new_x_ref,new_u_ref,new_lambda,new_mu = split_y(y_new)
        after_h_res = self.getCollisionResidual(new_x_ref)
        self.print_debug(f'after dyn correction before h_res = {apriori_h_res} -> after {after_h_res}')
        '''

        index = 0
        h_plus_violations = 0

        # FIXME debug
        self.print_debug(' r_0 breakdown ')
        for i in range(self.N):
            self.print_debug(f'agent {i}')
            dLL_dx_res = np.linalg.norm(r0[index:index+n*T])
            index += n*T
            dLL_du_res = np.linalg.norm(r0[index:index+m*T])
            index += m*T
            fx_res = np.linalg.norm(r0[index:index+n*T])
            index += n*T
            index += np.sum(h_plus_mask[:,i])
            self.print_debug(f'dLL_dx {dLL_dx_res:.4f}, dLL_du {dLL_du_res:.4f}, fx {fx_res:.8f}, h_plus {np.sum(h_plus_mask[:,i])}')

        #check different parts of the residuals, with rho = infty
        original_rho = self.rho
        #self.rho = 1e4
        index = 0
        h_plus_violations = 0
        # self.print_debug(f' r_t breakdown, step = {step} ')
        for i in range(self.N):
            self.print_debug(f'agent {i}')
            dLL_dx_res = np.linalg.norm(r_t[index:index+n*T])
            index += n*T
            dLL_du_res = np.linalg.norm(r_t[index:index+m*T])
            index += m*T
            fx_res = np.linalg.norm(r_t[index:index+n*T])
            index += n*T
            index += np.sum(h_plus_mask[:,i])
            self.print_debug(f'dLL_dx {dLL_dx_res:.4f}, dLL_du {dLL_du_res:.4f}, fx {fx_res:.8f}, h_plus {np.sum(h_plus_mask[:,i])}')
        self.rho = original_rho

        # FIXME debug, compare rollout vs current x_ref
        '''
        try:
            diff = self.rollout(self.x0, new_u_ref) - new_x_ref
            dyn_res = np.linalg.norm(diff)
            self.print_debug(f'dynamics residual {dyn_res}')
            r_diff = np.abs(self.old_rt - r0)>1e-8
        except AttributeError:
            pass
        self.old_rt = r_t.copy()
        self.old_y = y_new
        '''

        # FIXME debug
        self.residual_vec.append(r0_norm)
        violations = np.sum(h_plus_mask)/2
        expected_posterior_norm = np.linalg.norm(r0 + Dr @ dy)

        self.print_debug(f'r0_norm {r0_norm} expected full step {expected_posterior_norm} rt_norm {r_t_norm}, h>0 {violations}')

        # stopping criterion
        if (np.abs(r_t_norm)<self.tolerance and violations == 0):
            raise StopIteration('stopping criterion met!')
        if (flag_no_step):
            raise StopIteration('iteration not making progress')

        self.rho *= self.rho_b

        if (self.USE_CPP):
            # normally we won't reach here because we'd use  the cpp.step(),
            # but if we are only using the "subfunctions", then this will be called
            self.cpp.post_step_update()


        return split_y(y_new)

    def rollout(self,x0,U):
        ''' given x0 and u0..u_T-1 (T*N*m), find x1..xT '''
        U = U.reshape(self.T,self.N,self.m)
        X = np.zeros((self.T+1,self.N,self.n))
        X[0,:,:] = x0.reshape(self.N,self.n)
        # x+ = x + vx*dt + 0.5*ax*dt*dt
        # vx+ = vx + ax*dt
        for i in range(self.N):
            for k in range(1,self.T+1):
                X[k,i] = self.f(X[k-1,i], U[k-1,i],i)
        return X[1:,:,:]

    def visualize(self,U,X=None,visualize=False,save_gif=False,animate=False,gif_prefix='run'):
        if (visualize or save_gif):
            fig = self._visualize(U,X)
            if (save_gif):
                fig.canvas.draw()
                frame = Image.frombytes('RGB',
                fig.canvas.get_width_height(),fig.canvas.tostring_rgb())
                self.frame_vec.append(frame)
            if (visualize):
                plt.show()
        if (animate):
            self._animation(U,X,gif_prefix=gif_prefix)

        return

    def final(self):
        self.profiler.summary()
        if (self.USE_CPP):
            self.cpp.summary()
        if (len(self.frame_vec)>0):
            gif_filename = self.resolveLogname()
            self.frame_vec[0].save(fp=gif_filename,format='GIF',append_images=self.frame_vec,save_all=True,duration = 200,loop=0)
            self.print_debug(f'GIf saved to {gif_filename}')
        '''
        plt.plot(self.residual_vec,'*-')
        plt.yscale('log')
        plt.xlabel('Iteration')
        plt.ylabel('Residual (exp)')
        plt.show()
        '''

    def resolveLogname(self,logPrefix='run'):
        # setup log file
        # log file will record state of the vehicle for later analysis
        logFolder = "./gifs/"
        logSuffix = ".gif"
        no = 1
        while os.path.isfile(logFolder+logPrefix+str(no)+logSuffix):
            no += 1

        log_no = no
        logFilename = logFolder+logPrefix+str(no)+logSuffix
        return logFilename

    def dr_dy(self, x, u, lamda, mu, h_plus_mask):
        t = self.profiler
        if (self.USE_CPP):
            '''
            t.s('drdy-stacked')
            drdx = self.cpp.dr_dx(x, u, lamda, mu, h_plus_mask)
            drdu = self.cpp.dr_du(x, u, lamda, mu, h_plus_mask)
            drdlamda = self.cpp.dr_dlamda(x, u, lamda, mu, h_plus_mask)
            drdmu = self.cpp.dr_dmu(x, u, lamda, mu, h_plus_mask)
            Dr = np.hstack([drdx,drdu,drdlamda,drdmu])
            t.e('drdy-stacked')
            '''
            t.s('drdy-cpp')
            Dr = self.cpp.dr_dy(x, u, lamda, mu, h_plus_mask)
            t.e('drdy-cpp')
        else:
            t.s('drdx')
            drdx = self.dr_dx(x, u, lamda, mu, h_plus_mask)
            t.e('drdx')
            t.s('drdu')
            drdu = self.dr_du(x, u, lamda, mu, h_plus_mask)
            t.e('drdu')
            t.s('drdlamda')
            drdlamda = self.dr_dlamda(x, u, lamda, mu, h_plus_mask)
            t.e('drdlamda')
            t.s('drdmu')
            drdmu = self.dr_dmu(x, u, lamda, mu, h_plus_mask)
            t.e('drdmu')
            t.s('stack')
            Dr = np.hstack([drdx,drdu,drdlamda,drdmu])
            t.e('stack')
        return Dr

    def getCollisionResidual(self, x):
        h_res = 0
        for k in range(1,self.T+1):
            for i in range(self.N):
                for j in range(i+1,self.N):
                    this_h = self.h(x[k-1,i],x[k-1,j])
                    if (this_h > 0):
                        h_res += this_h
        return h_res

    def getHplusMask(self,x):
        # h(i,i) should not be considered in either h_plus or h_minus
        # we check it in h_minux
        # for x 1-T, NOTE index start from 1
        h_plus_mask = np.zeros((self.T,self.N,self.N),dtype=bool)
        for k in range(1,self.T+1):
            for i in range(self.N):
                for j in range(i+1,self.N):
                    h_plus_mask[k-1,i,j] = h_plus_mask[k-1,j,i] = self.h(x[k-1,i],x[k-1,j]) >= 0
        if (self.CPP_DEBUG):
            alt = self.cpp.getHplusMask([xx for xx in x])
            if (np.linalg.norm(alt-h_plus_mask)>1e-4):
                breakpoint()

        return h_plus_mask
    # ----- derivatives and other generic math functions ----
    def L(self,x_k, u_k_i, x_k1_i, h_k_plus_mask,lamda_k, mu_k,i):
        # feasibility for h>0
        h_plus = np.sum( [ mu_k[i,j.item()] * ( self.h(x_k[i], x_k[j.item()]) ) for j in np.nonzero(h_k_plus_mask[i])[0] ],axis=0)
        if (self.CPP_DEBUG):
            for j in np.nonzero(h_k_plus_mask[i])[0]:
                val = self.h(x_k[i], x_k[j.item()])
                val_cpp = self.cpp.h(x_k[i], x_k[j.item()])
                if (np.linalg.norm(val-val_cpp)>1e-4):
                    breakpoint()

        # barrier for h < 0
        h_minus = -1/self.rho*np.sum([np.log(-min(self.h(x_k[i], x_k[j.item()]),-1e-100)) if j.item() != i else 0 for j in np.nonzero(~h_k_plus_mask[i])[0] ])
        dynamics = lamda_k[i].T @ ( self.f(x_k[i],u_k_i,i) - x_k1_i)
        return self.J(x_k, u_k_i, i) + h_plus + h_minus + dynamics

    def dL_dx_ik(self,x_k, u_k_i, x_k1_i, h_k_plus_mask,lamda_k, mu_k,i):
        if (self.USE_CPP):
            return self.cpp.dL_dx_ik(x_k, u_k_i, x_k1_i, h_k_plus_mask,lamda_k, mu_k,i)

        val =  self.dJi_dxi(x_k,u_k_i,i) + lamda_k[i].T @ self.df_dx(x_k[i],u_k_i,i)
        val += np.sum( [ mu_k[i,j.item()] * ( self.dh_dxi(x_k[i], x_k[j.item()]) ) for j in np.nonzero(h_k_plus_mask[i])[0] ], axis=0)
        val += -1.0/self.rho*np.sum([min(1/self.h(x_k[i], x_k[j.item()]),1e20) * self.dh_dxi(x_k[i], x_k[j.item()]) * (j.item() != i) for j in np.nonzero(~h_k_plus_mask[i])[0] ],axis=0)
        #h_val_alt = np.sum([ self.dBh_dxi(x_k[i], x_k[j.item()]) * (j.item()!=i) for j in np.nonzero(~h_k_plus_mask[i])[0] ],axis=0)

        # NOTE the behavior of barrier function near boundary may need tuning
        if (self.DEBUG):
            # dJi_dx -- passed
            num = jacobianNumerical(lambda xx:self.J(xx.reshape(x_k.shape),u_k_i,i), x_k.flatten()).reshape(1,self.N,self.n)[:,i]
            ana = self.dJi_dxi(x_k, u_k_i, i)
            assert (np.linalg.norm(num-ana) < 1e-4)
            # df_dx -- inconclusive
            num = jacobianNumerical(lambda uu:self.J(x_k,uu,i), u_k_i)
            ana = self.dJi_du(x_k, u_k_i, i)
            assert (np.linalg.norm(num-ana) < 1e-4)
            # dh_dxi -- inconclusive
            for j in np.nonzero(h_k_plus_mask[i])[0]:
                if i == j:
                    continue
                ana = self.dh_dxi(x_k[i], x_k[j])
                num = jacobianNumerical(lambda xx:self.h(xx,x_k[j]),x_k[i])
                assert (np.linalg.norm(num-ana) < 1e-4)

            num = jacobianNumerical(lambda xx:self.L(xx.reshape(x_k.shape), u_k_i, x_k1_i, h_k_plus_mask,lamda_k, mu_k,i),x_k.flatten())
            num = num[0,i*self.n:(i+1)*self.n]
            assert (np.linalg.norm(num-val) < 1e-4)

        if (self.CPP_DEBUG):
            alt = self.cpp.dJi_dxi(x_k, u_k_i, i)
            if (np.linalg.norm(self.dJi_dxi(x_k,u_k_i,i)-alt)>1e-4):
                breakpoint()
            alt = self.cpp.dL_dx_ik(x_k, u_k_i, x_k1_i, h_k_plus_mask,lamda_k, mu_k,i)
            if (np.linalg.norm(val-alt)>1e-4):
                breakpoint()
        return val

    def dL_dx_ik1(self,x_k, u_k_i, x_k1_i, h_k_plus_mask,lamda_k, mu_k,i):
        ''' dL/dx_i_k+1 '''
        return -lamda_k[i].T

    # NOTE obsolete only needed in dLLi_dxj, which is obsolete
    def dL_dx_jk(self,x_k, u_k_i, x_k1_i, h_k_plus_mask,lamda_k, mu_k,i,j):
        assert (i!=j)
        return  self.dJi_dxj(x[k-1],u[k,i],i,j) + ( mu_k[i,j] * ( self.dh_dxj(x_k[i], x_k[j]) ) if h_k_plus_mask[i,j] else \
            -1/self.rho*min(1/self.h(x_k[i], x_k[j]),1e20) * self.dh_dxi(x_k[i], x_k[j]) )

    # NOTE deprecated, usually dJi_du is called directly
    def dL_du(self,x_k, u_k_i, x_k1_i, h_k_plus_mask,lamda_k, mu_k,i):
        val = self.dJi_du(x_k,u_k_i,i) + lamda_k[i].T @ self.df_du(x_k[i], u_k_i,i)
        return val


    # only used in debug, LLi's derivative is used more prevalently
    def LLi(self,x,u,h_plus_mask,lamda,mu,i):
        T = self.T
        LLi_val = np.sum( [self.L(x[k-1],u[k,i], x[k,i],h_plus_mask[k-1], lamda[k], mu[k-1],i) for k in range(1,T)] ,axis=0)
        # x0 related terms
        LLi_val += self.J(self.x0,u[0,i],i) + lamda[0,i].T @ ( self.f(self.x0[i],u[0,i],i) - x[0,i])
        # x_T related terms
        LLi_val += self.Jfi(x[T-1],i)
        h_plus = np.sum( [ mu[T-1,i,j.item()] * ( self.h(x[T-1,i], x[T-1,j.item()]) ) for j in np.nonzero(h_plus_mask[T-1,i])[0] ],axis=0)
        h_minus = -1/self.rho*np.sum([np.log(-min(self.h(x[T-1,i], x[T-1,j.item()]),-1e-100)) * (j.item() != i) for j in np.nonzero(~h_plus_mask[T-1,i])[0] ],axis=0)
        LLi_val += h_plus + h_minus
        return LLi_val

    def dLLi_dxi(self,x,u,h_plus_mask,lamda,mu,i):
        if (self.USE_CPP):
            return self.cpp.dLLi_dxi([xx for xx in x],[uu for uu in u],[hh for hh in h_plus_mask],[ll for ll in lamda],[mmm for mmm in mu],i)
        ''' return: 1*(T*n)  Note index of x starts with 1'''
        T = self.T; N = self.N; n = self.n; m = self.m
        der = np.zeros(T*n)
        submtx_k = lambda k:der[(k-1)*n:k*n]
        # dLLi_dxi
        for k in range(1,T):
            sub = submtx_k(k)
            sub[:] = self.dL_dx_ik(x[k-1],u[k,i],x[k,i],h_plus_mask[k-1],lamda[k],mu[k-1],i) -lamda[k-1,i].T
            if (self.DEBUG):
                num = jacobianNumerical(lambda xx:self.L(xx.reshape(N,n), u[k,i], x[k,i], h_plus_mask[k-1],lamda[k], mu[k-1],i),x[k-1].flatten())
                num = num[0,i*n:(i+1)*n] - lamda[k-1,i].T
                assert( np.linalg.norm(num-sub) < 1e-4)

        # dLLi_dxi_T
        sub = submtx_k(T)
        sub[:] = -lamda[T-1,i].T + self.dJfi_dxi(x[T-1],i) \
            + np.sum( [ mu[T-1,i,j.item()] * ( self.dh_dxi(x[T-1,i], x[T-1,j.item()]) ) for j in np.nonzero(h_plus_mask[T-1,i])[0] ],axis=0) \
            -1/self.rho*np.sum([min(1/self.h(x[T-1,i], x[T-1,j.item()]),1e20)*self.dh_dxi(x[T-1,i],x[T-1,j.item()]) *(j.item() != i) for j in np.nonzero(~h_plus_mask[T-1,i])[0] ],axis=0)

        val = der.reshape(1,-1)
        if (self.CPP_DEBUG):
            alt = self.cpp.dLLi_dxi([xx for xx in x],[uu for uu in u],[hh for hh in h_plus_mask],[ll for ll in lamda],[mmm for mmm in mu],i)
            if (np.linalg.norm(alt-val)>1e-4):
                breakpoint()
        return val

    # NOTE obsolete, now we use dLLi_dxi
    def dLLi_dx(self,x,u,h_plus_mask,lamda,mu,i):
        if (self.USE_CPP):
            return self.cpp.dLLi_dx([xx for xx in x],[uu for uu in u],[hh for hh in h_plus_mask],[ll for ll in lamda],[mmm for mmm in mu],i)
        ''' return: 1*dim(x) = 1*(T*N*n) , Note index of x starts with 1'''
        T = self.T; N = self.N; n = self.n; m = self.m
        der = np.zeros(T*N*n)
        submtx_i_k = lambda i,k:der[(k-1)*N*n+i*n:(k-1)*N*n+(i+1)*n]
        # dLLi_dxi
        for k in range(1,T):
            sub = submtx_i_k(i,k)
            sub[:] = self.dL_dx_ik(x[k-1],u[k,i],x[k,i],h_plus_mask[k-1],lamda[k],mu[k-1],i) -lamda[k-1,i].T
            if (self.DEBUG):
                num = jacobianNumerical(lambda xx:self.L(xx.reshape(N,n), u[k,i], x[k,i], h_plus_mask[k-1],lamda[k], mu[k-1],i),x[k-1].flatten())
                num = num[0,i*n:(i+1)*n] - lamda[k-1,i].T
                assert( np.linalg.norm(num-sub) < 1e-4)

        # dLLi_dxi_T
        sub = submtx_i_k(i,T)
        sub[:] = -lamda[T-1,i].T + self.dJfi_dxi(x[T-1],i) \
            + np.sum( [ mu[T-1,i,j.item()] * ( self.dh_dxi(x[T-1,i], x[T-1,j.item()]) ) for j in np.nonzero(h_plus_mask[T-1,i])[0] ],axis=0) \
            -1/self.rho*np.sum([min(1/self.h(x[T-1,i], x[T-1,j.item()]),1e20)*self.dh_dxi(x[T-1,i],x[T-1,j.item()]) *(j.item() != i) for j in np.nonzero(~h_plus_mask[T-1,i])[0] ],axis=0)

        # dLLi_dxj
        for j in range(0,N):
            if i==j:
                continue
            for k in range(1,T):
                sub = submtx_i_k(j,k)
                sub[:] = self.dJi_dxj(x[k-1],u[k,i],i,j) + (mu[k-1,i,j] * self.dh_dxj(x[k-1,i], x[k-1,j]) if h_plus_mask[k-1,i,j] else \
                    -1/self.rho*min(1/self.h(x[k-1,i], x[k-1,j]),2e10)*self.dh_dxj(x[k-1,i], x[k-1,j]))
            k = T
            sub = submtx_i_k(j,k)
            sub[:] = self.dJfi_dxj(x[k-1],i,j) + (mu[k-1,i,j] * self.dh_dxj(x[k-1,i], x[k-1,j]) if h_plus_mask[k-1,i,j] else \
                -1/self.rho*min(1/self.h(x[k-1,i], x[k-1,j]),2e10)*self.dh_dxj(x[k-1,i], x[k-1,j]))
        val = der.reshape(1,-1)
        if (self.CPP_DEBUG):
            alt = self.cpp.dLLi_dx([xx for xx in x],[uu for uu in u],[hh for hh in h_plus_mask],[ll for ll in lamda],[mmm for mmm in mu],i)
            if (np.linalg.norm(alt-val)>1e-4):
                breakpoint()
        return val

    def dLLi_dui(self,x,u,h_plus_mask,lamda,mu,i):
        if (self.USE_CPP):
            return self.cpp.dLLi_dui([xx for xx in x],[uu for uu in u],[hh for hh in h_plus_mask],[ll for ll in lamda],[mmm for mmm in mu],i)
        ''' return: 1*(T*m) '''
        T = self.T; N = self.N; n = self.n; m = self.m
        der = np.zeros(T*m)
        submtx_k = lambda k:der[k*m:(k+1)*m]
        # dLLi_dui_0
        sub = submtx_k(0)
        sub[:] = self.dJi_du(self.x0,u[0,i],i) + lamda[0,i].T @ self.df_du(self.x0[i],u[0,i],i)
        # dLLi_dui_k
        for k in range(1,T):
            sub = submtx_k(k)
            # dL_du
            sub[:] = self.dJi_du(x[k-1],u[k,i],i) + lamda[k,i].T @ self.df_du(x[k-1,i],u[k,i],i)
        val = der.reshape(1,-1)
        if (self.CPP_DEBUG):
            alt = self.cpp.dLLi_dui([xx for xx in x],[uu for uu in u],[hh for hh in h_plus_mask],[ll for ll in lamda],[mmm for mmm in mu],i)
            if (np.linalg.norm(alt-der)>1e-4):
                breakpoint()
        return der

    # NOTE obsolete
    def dLLi_du(self,x,u,h_plus_mask,lamda,mu,i):
        if (self.USE_CPP):
            return self.cpp.dLLi_du([xx for xx in x],[uu for uu in u],[hh for hh in h_plus_mask],[ll for ll in lamda],[mmm for mmm in mu],i)
        ''' return: 1*dim(u) = 1*(T*N*m) '''
        T = self.T; N = self.N; n = self.n; m = self.m
        der = np.zeros(T*N*m)
        submtx_i_k = lambda i,k:der[k*N*m+i*m:k*N*m+(i+1)*m]
        # dLLi_dui_0
        sub = submtx_i_k(i,0)
        sub[:] = self.dJi_du(self.x0,u[0,i],i) + lamda[0,i].T @ self.df_du(self.x0[i],u[0,i],i)
        # dLLi_dui_k
        for k in range(1,T):
            sub = submtx_i_k(i,k)
            # dL_du
            sub[:] = self.dJi_du(x[k-1],u[k,i],i) + lamda[k,i].T @ self.df_du(x[k-1,i],u[k,i],i)
        if (self.CPP_DEBUG):
            alt = self.cpp.dLLi_du([xx for xx in x],[uu for uu in u],[hh for hh in h_plus_mask],[ll for ll in lamda],[mmm for mmm in mu],i)
            if (np.linalg.norm(alt-der)>1e-4):
                breakpoint()
        return der

    def dLLi_dxi_dmu(self,x,u,h_plus_mask,lamda,mu,i):
        if (self.USE_CPP):
            return self.cpp.dLLi_dxi_dmu([xx for xx in x],[uu for uu in u],[hh for hh in h_plus_mask],[ll for ll in lamda],[mmm for mmm in mu],i)
        ''' return: dim: dim_x*dim_mu '''
        T = self.T; N = self.N; n = self.n; m = self.m
        dim_x = T*N*n; dim_u = T*N*m
        dim_mu = T*N*N
        dLL_dxi_dmu = np.zeros((T*n,dim_mu))
        for k in range(1,T+1):
            for j in np.nonzero(h_plus_mask[k-1,i])[0]:
                dLLi_dxki_dmuijk = self.dh_dxi(x[k-1,i],x[k-1,j])
                dLL_dxi_dmu[(k-1)*n:k*n, (k-1)*N*N+i*N+j] = dLLi_dxki_dmuijk
        if (self.CPP_DEBUG):
            alt = self.cpp.dLLi_dxi_dmu([xx for xx in x],[uu for uu in u],[hh for hh in h_plus_mask],[ll for ll in lamda],[mmm for mmm in mu],i)
            if (np.linalg.norm(alt-dLL_dx_dmu)>1e-4):
                breakpoint()
        return dLL_dxi_dmu

    # NOTE obsolete, use dLLi_dxi_dmu now
    def dLLi_dx_dmu(self,x,u,h_plus_mask,lamda,mu,i):
        if (self.USE_CPP):
            return self.cpp.dLLi_dx_dmu([xx for xx in x],[uu for uu in u],[hh for hh in h_plus_mask],[ll for ll in lamda],[mmm for mmm in mu],i)
        ''' return: dim: dim_x*dim_mu '''
        T = self.T; N = self.N; n = self.n; m = self.m
        dim_x = T*N*n; dim_u = T*N*m
        dim_mu = T*N*N
        dLL_dx_dmu = np.zeros((dim_x,dim_mu))
        for k in range(1,T+1):
            for j in np.nonzero(h_plus_mask[k-1,i])[0]:
                dLLi_dxki_dmuijk = self.dh_dxi(x[k-1,i],x[k-1,j])
                dLLi_dxkj_dmuijk = self.dh_dxj(x[k-1,i],x[k-1,j])
                dLL_dx_dmu[(k-1)*N*n+i*n:(k-1)*N*n+(i+1)*n,(k-1)*N*N+i*N+j] = dLLi_dxki_dmuijk
                dLL_dx_dmu[(k-1)*N*n+j*n:(k-1)*N*n+(j+1)*n,(k-1)*N*N+i*N+j] = dLLi_dxkj_dmuijk
        if (self.CPP_DEBUG):
            alt = self.cpp.dLLi_dx_dmu([xx for xx in x],[uu for uu in u],[hh for hh in h_plus_mask],[ll for ll in lamda],[mmm for mmm in mu],i)
            if (np.linalg.norm(alt-dLL_dx_dmu)>1e-4):
                breakpoint()
        return dLL_dx_dmu


    '''
    def r_numerical(self, x, u, lamda, mu, h_plus_mask):
        T = self.T
        try:
            r = np.zeros(0)
            for i in range(self.N):
                dLL_dx = jacobianNumerical(lambda xx:self.LLi(xx.reshape(x.shape),u,h_plus_mask,lamda,mu,i), x.flatten())
                dLL_du = jacobianNumerical(lambda uu:self.LLi(x,uu.reshape(u.shape),h_plus_mask,lamda,mu,i), u.flatten())
                r = np.hstack([r,dLL_dx.flatten(), dLL_du.flatten()])
                # dynamics for f(x0,u0) = x1
                r = np.hstack([r,self.f(self.x0[i], u[0,i],i) - x[0,i]])
                for k in range(1,self.T):
                    r = np.hstack([r,self.f(x[k-1,i], u[k,i],i) - x[k,i]]) # dual for dynamics
                    r = np.hstack([r]+[ self.h(x[k-1,i], x[k-1,j.item()]) for j in np.nonzero(h_plus_mask[k-1,i])[0] ])
                # h(x_T_i, x_T_j)
                r = np.hstack([r]+[ self.h(x[T-1,i], x[T-1,j.item()]) for j in np.nonzero(h_plus_mask[T-1,i])[0] ])
        except ValueError as e:
            raise e
            breakpoint()
        return r
    '''

    def r(self, x, u, lamda, mu, h_plus_mask):
        if (self.USE_CPP):
            return self.cpp.r([xx for xx in x],[uu for uu in u],[ll for ll in lamda],[mmm for mmm in mu],[hh for hh in h_plus_mask])
        T = self.T
        try:
            r = np.zeros(0)
            for i in range(self.N):
                dLL_dxi = self.dLLi_dxi(x,u,h_plus_mask,lamda,mu,i)
                dLL_dui = self.dLLi_dui(x,u,h_plus_mask,lamda,mu,i)
                '''
                # this needs to be updated
                if (self.DEBUG):
                    dLL_du_num = jacobianNumerical(lambda uu:self.LLi(x,uu.reshape(u.shape),h_plus_mask,lamda,mu,i), u.flatten())
                    assert(np.linalg.norm(dLL_du-dLL_du_num)<1e-4)
                    dLL_dx_num = jacobianNumerical(lambda xx:self.LLi(xx.reshape(x.shape),u,h_plus_mask,lamda,mu,i), x.flatten())
                    assert(np.linalg.norm(dLL_dx-dLL_dx_num)<1e-4)
                '''
                r = np.hstack([r,dLL_dxi.flatten(), dLL_dui.flatten()])
                # dynamics for f(x0,u0) = x1
                r = np.hstack([r, self.dynamics_residual_weight * self.f(self.x0[i], u[0,i],i) - x[0,i]])
                for k in range(1,self.T):
                    r = np.hstack([r, self.dynamics_residual_weight * self.f(x[k-1,i], u[k,i],i) - x[k,i]]) # dual for dynamics
                for k in range(1,self.T):
                    r = np.hstack([r]+[ self.h(x[k-1,i], x[k-1,j.item()]) for j in np.nonzero(h_plus_mask[k-1,i])[0] ])
                # h(x_T_i, x_T_j)
                r = np.hstack([r]+[ self.h(x[T-1,i], x[T-1,j.item()]) for j in np.nonzero(h_plus_mask[T-1,i])[0] ])
        except ValueError as e:
            raise e
            breakpoint()

        if (self.CPP_DEBUG):
            alt = self.cpp.r([xx for xx in x],[uu for uu in u],[ll for ll in lamda],[mmm for mmm in mu],[hh for hh in h_plus_mask])
            if (np.linalg.norm(alt.flatten()-r)>1e-4):
                breakpoint()
        return r

    '''
    def r_fillin(self, x, u, lamda, mu, h_plus_mask):
        T = self.T; N = self.N; n = self.n; m = self.m
        dim_x = N*T*n; dim_u = N*T*m
        dim_r = N*(dim_x+dim_u+T*n)+np.sum(h_plus_mask)
        try:
            r = np.zeros(dim_r)
            index = 0
            for i in range(self.N):
                dLL_dx = self.dLLi_dx(x,u,h_plus_mask,lamda,mu,i)
                dLL_du = self.dLLi_du(x,u,h_plus_mask,lamda,mu,i)
                if (self.DEBUG):
                    dLL_du_num = jacobianNumerical(lambda uu:self.LLi(x,uu.reshape(u.shape),h_plus_mask,lamda,mu,i), u.flatten())
                    assert(np.linalg.norm(dLL_du-dLL_du_num)<1e-4)
                    dLL_dx_num = jacobianNumerical(lambda xx:self.LLi(xx.reshape(x.shape),u,h_plus_mask,lamda,mu,i), x.flatten())
                    assert(np.linalg.norm(dLL_dx-dLL_dx_num)<1e-4)
                r[index:index+dim_x] = dLL_dx.flatten()
                index += dim_x
                r[index:index+dim_u] = dLL_du.flatten()
                index += dim_u

                # dynamics for f(x0,u0) = x1
                r[index: index+n] = self.f(self.x0[i], u[0,i],i) - x[0,i]
                for k in range(1,self.T):
                    r[index+k*n: index+(k+1)*n] = self.f(x[k-1,i], u[k,i],i) - x[k,i] # dual for dynamics
                index += n*T
                for k in range(1,self.T+1):
                    indices = np.nonzero(h_plus_mask[k-1,i])[0]
                    if (len(indices) == 0):
                        continue
                    hh = np.hstack([ self.h(x[k-1,i], x[k-1,j.item()]) for j in indices ])
                    r[index: index+hh.shape[0]] = hh
                    index += hh.shape[0]
        except ValueError as e:
            raise e
            breakpoint()
        return r
    '''

    def Bh(self,x_i,x_j):
        return -1/self.rho * np.log(-min(self.h(x_i, x_j),-1e-100))

    def dBh_dxi(self,x_i,x_j):
        if (self.USE_CPP):
            return self.cpp.dBh_dxi(x_i,x_j)
        # B(h) = -rho^-1 log(-h)
        #dB(h)/dx = -rho^-1 h^-1 dhdx
        val = -1/(self.rho*self.h(x_i, x_j))* self.dh_dxi(x_i,x_j)
        if (self.DEBUG):
            val_num = jacobianNumerical(lambda xx:self.Bh(xx.reshape(x_i.shape),x_j), x_i.flatten())
            assert(np.linalg.norm(val-val_num)<1e-4)
        if (self.CPP_DEBUG):
            alt = self.cpp.dBh_dxi(x_i,x_j)
            if (np.linalg.norm(alt-val)>1e-4):
                breakpoint()
        return val

    def dBh_dxj(self,x_i,x_j):
        if (self.USE_CPP):
            return self.cpp.dBh_dxj(x_i,x_j)
        # B(h) = -rho^-1 log(-h)
        #dB(h)/dx = -rho^-1 h^-1 dhdx
        val = -1/(self.rho*self.h(x_i, x_j))* self.dh_dxj(x_i,x_j)
        if (self.DEBUG):
            val_num = jacobianNumerical(lambda xx:self.Bh(x_i,xx.reshape(x_j.shape)), x_j.flatten())
            assert(np.linalg.norm(val-val_num)<1e-4)
        if (self.CPP_DEBUG):
            alt = self.cpp.dBh_dxj(x_i,x_j)
            if (np.linalg.norm(alt-val)>1e-4):
                breakpoint()
        return val

    def dBh_dxi_dxi(self,x_i,x_j):
        if (self.USE_CPP):
            return self.cpp.dBh_dxi_dxi(x_i,x_j)
        h = self.h(x_i,x_j)
        dhdxi = self.dh_dxi(x_i,x_j).reshape(1,self.n)
        val = 1/(self.rho * h) * (-self.dh_dxi_dxi(x_i,x_j) + 1/h * dhdxi.T @ dhdxi )
        if (self.DEBUG):
            val_num = jacobianNumerical(lambda xx:self.dBh_dxi(xx.reshape(x_i.shape),x_j), x_i.flatten(),dim=self.n)
            assert(np.linalg.norm(val-val_num)<1e-4)
        if (self.CPP_DEBUG):
            alt = self.cpp.h(x_i,x_j)
            if (np.linalg.norm(alt-h)>1e-4):
                breakpoint()
            alt = self.cpp.dh_dxi(x_i,x_j)
            if (np.linalg.norm(alt-dhdxi)>1e-4):
                breakpoint()
            dhii = self.dh_dxi_dxi(x_i,x_j)
            alt = self.cpp.dh_dxi_dxi(x_i,x_j)
            if (np.linalg.norm(alt-dhii)>1e-4):
                breakpoint()
            alt = self.cpp.dBh_dxi_dxi(x_i,x_j)
            if (np.linalg.norm(alt-val)>1e-4):
                breakpoint()

        return val

    def dBh_dxi_dxj(self,x_i,x_j):
        if (self.USE_CPP):
            return self.cpp.dBh_dxi_dxj(x_i,x_j)
        h = self.h(x_i,x_j)
        dhdxi = self.dh_dxi(x_i,x_j).reshape(1,self.n)
        dhdxj = self.dh_dxj(x_i,x_j).reshape(1,self.n)
        val = 1/(self.rho * h) * (-self.dh_dxi_dxj(x_i,x_j) + 1/h * dhdxi.T @ dhdxj )
        if (self.DEBUG):
            val_num = jacobianNumerical(lambda xx:self.dBh_dxi(x_i, xx.reshape(x_j.shape)), x_j.flatten(),dim=self.n)
            assert(np.linalg.norm(val-val_num)<1e-4)
        if (self.CPP_DEBUG):
            alt = self.cpp.dh_dxi(x_i,x_j)
            if (np.linalg.norm(alt-dhdxi)>1e-4):
                breakpoint()
            alt = self.cpp.dh_dxj(x_i,x_j)
            if (np.linalg.norm(alt-dhdxj)>1e-4):
                breakpoint()
            alt = self.cpp.dBh_dxi_dxj(x_i,x_j)
            if (np.linalg.norm(alt-val)>1e-4):
                breakpoint()

        return val

    def dBh_dxj_dxj(self,x_i,x_j):
        if (self.USE_CPP):
            return self.cpp.dBh_dxj_dxj(x_i,x_j)
        h = self.h(x_i,x_j)
        dhdxi = self.dh_dxi(x_i,x_j).reshape(1,self.n)
        dhdxj = self.dh_dxj(x_i,x_j).reshape(1,self.n)
        val = 1/(self.rho * h) * (-self.dh_dxj_dxj(x_i,x_j) + 1/h * dhdxj.T @ dhdxj )
        if (self.DEBUG):
            val_num = jacobianNumerical(lambda xx:self.dBh_dxj(x_i, xx.reshape(x_j.shape)), x_j.flatten(),dim=self.n)
            assert(np.linalg.norm(val-val_num)<1e-4)
        if (self.CPP_DEBUG):
            alt = self.cpp.dBh_dxj_dxj(x_i,x_j)
            if (np.linalg.norm(alt-val)>1e-4):
                breakpoint()
        return val

    def dLLi_dxi_dx(self,x,u,h_plus_mask,lamda,mu,i):
        if (self.USE_CPP):
            return self.cpp.dLLi_dxi_dx([xx for xx in x],[uu for uu in u],[hh for hh in h_plus_mask],[ll for ll in lamda],[mm for mm in mu],i)
        T = self.T; N = self.N; n = self.n; m = self.m; dim_x = T*N*n
        dLL_dxi_dx = np.zeros((T*n,dim_x))
        submtx = lambda k,j: dLL_dxi_dx[(k-1)*n:k*n, (k-1)*N*n+j*n:(k-1)*N*n+(j+1)*n]
        submtx_num = lambda k,j: dLL_dxi_dx_num[(k-1)*n:k*n, (k-1)*N*n+j*n:(k-1)*N*n+(j+1)*n]
        for k in range(1,T):
            #dLLi_dxki_dxki
            mtx = submtx(k,i)
            val1 = self.dJi_dxi_dxi(x[k-1],u[k,i],i)
            val2 = np.sum( [ mu[k-1,i,j.item()] * ( self.dh_dxi_dxi(x[k-1,i], x[k-1,j.item()]) ) for j in np.nonzero(h_plus_mask[k-1,i])[0] ],axis=0)
            val3 = np.sum([self.dBh_dxi_dxi(x[k-1,i], x[k-1,j.item()]) * (j.item() != i) for j in np.nonzero(~h_plus_mask[k-1,i])[0] ],axis=0)
            mtx[:,:] = val1 + val2 + val3

        #dLLi_dxki_dxki, k=T, u_T is undefined, use 0 to penalize J(x) only
        mtx = submtx(T,i)
        mtx[:,:] = self.dJfi_dxi_dxi(x[T-1],i) \
                + np.sum( [ mu[T-1,i,j.item()] * ( self.dh_dxi_dxi(x[T-1,i], x[T-1,j.item()]) ) for j in np.nonzero(h_plus_mask[T-1,i])[0] ],axis=0) \
                + np.sum([self.dBh_dxi_dxi(x[T-1,i], x[T-1,j.item()]) * (j.item() != i) for j in np.nonzero(~h_plus_mask[T-1,i])[0] ],axis=0)

        # dLLi_dxi_dxj
        for k in range(1,T):
            for j in range(N):
                if i==j:
                    continue
                if (j in np.nonzero(h_plus_mask[k-1,i])[0]):
                    #dLLi_dxki_dxkj
                    val = self.dJi_dxi_dxj(x[k-1],u[k,i],i,j) + mu[k-1,i,j] * self.dh_dxi_dxj(x[k-1,i], x[k-1,j])
                    mtx = submtx(k,j)
                    mtx[:,:] = val
                else:
                    #dLLi_dxki_dxkj
                    val = self.dJi_dxi_dxj(x[k-1],u[k,i],i,j) + self.dBh_dxi_dxj(x[k-1,i], x[k-1,j])
                    mtx = submtx(k,j)
                    mtx[:,:] = val
        k = T
        for j in range(N):
            if i==j:
                continue
            if (j in np.nonzero(h_plus_mask[k-1,i])[0]):
                #dLLi_dxki_dxkj
                val = self.dJfi_dxi_dxj(x[k-1],i,j) + mu[k-1,i,j] * self.dh_dxi_dxj(x[k-1,i], x[k-1,j])
                mtx = submtx(k,j)
                mtx[:,:] = val
            else:
                #dLLi_dxki_dxkj
                val = self.dJfi_dxi_dxj(x[k-1],i,j) + self.dBh_dxi_dxj(x[k-1,i], x[k-1,j])
                mtx = submtx(k,j)
                mtx[:,:] = val

        if (self.DEBUG):
            '''
            for k in range(1,T+1):
                for ii in range(N):
                    for j in range(N):
                        mtx_num = submtx_num(k,ii,j)
                        mtx = submtx(k,ii,j)
                        if(not np.linalg.norm(mtx-mtx_num)<1e-4):
                            self.print_debug(f'i = {i} ii={ii},j={j},k={k}')
                            #breakpoint()
            '''
            dLL_dxi_dx_num = jacobianNumerical(lambda xx:self.dLLi_dxi(xx.reshape(x.shape),u,h_plus_mask,lamda,mu,i), x.flatten(),dim=dim_x)
            self.print_debug(f'dLL_dxdx err {np.linalg.norm(dLL_dxdx_num - dLL_dxdx)}')
            assert(np.linalg.norm(dLL_dxdx_num - dLL_dxdx)<1e-4)
        if (self.CPP_DEBUG):
            alt = self.cpp.dLLi_dxi_dx([xx for xx in x],[uu for uu in u],[hh for hh in h_plus_mask],[ll for ll in lamda],[mm for mm in mu],i)
            if (np.linalg.norm(alt-dLL_dxdx)>1e-4):
                breakpoint()
        return dLL_dxi_dx

    # NOTE obsolete
    def dLLi_dxdx(self,x,u,h_plus_mask,lamda,mu,i):
        if (self.USE_CPP):
            return self.cpp.dLLi_dxdx([xx for xx in x],[uu for uu in u],[hh for hh in h_plus_mask],[ll for ll in lamda],[mm for mm in mu],i)
        T = self.T; N = self.N; n = self.n; m = self.m; dim_x = T*N*n
        dLL_dxdx = np.zeros((dim_x,dim_x))
        submtx = lambda k,i,j: dLL_dxdx[(k-1)*N*n+i*n:(k-1)*N*n+(i+1)*n,(k-1)*N*n+j*n:(k-1)*N*n+(j+1)*n]
        submtx_num = lambda k,i,j: dLL_dxdx_num[(k-1)*N*n+i*n:(k-1)*N*n+(i+1)*n,(k-1)*N*n+j*n:(k-1)*N*n+(j+1)*n]
        for k in range(1,T):
            #dLLi_dxki_dxki
            mtx = submtx(k,i,i)
            val1 = self.dJi_dxi_dxi(x[k-1],u[k,i],i)
            val2 = np.sum( [ mu[k-1,i,j.item()] * ( self.dh_dxi_dxi(x[k-1,i], x[k-1,j.item()]) ) for j in np.nonzero(h_plus_mask[k-1,i])[0] ],axis=0)
            val3 = np.sum([self.dBh_dxi_dxi(x[k-1,i], x[k-1,j.item()]) * (j.item() != i) for j in np.nonzero(~h_plus_mask[k-1,i])[0] ],axis=0)
            mtx[:,:] = val1 + val2 + val3

        #dLLi_dxki_dxki, k=T, u_T is undefined, use 0 to penalize J(x) only
        mtx = submtx(T,i,i)
        mtx[:,:] = self.dJfi_dxi_dxi(x[T-1],i) \
                + np.sum( [ mu[T-1,i,j.item()] * ( self.dh_dxi_dxi(x[T-1,i], x[T-1,j.item()]) ) for j in np.nonzero(h_plus_mask[T-1,i])[0] ],axis=0) \
                + np.sum([self.dBh_dxi_dxi(x[T-1,i], x[T-1,j.item()]) * (j.item() != i) for j in np.nonzero(~h_plus_mask[T-1,i])[0] ],axis=0)

        # dLLi_dxi_dxj
        for k in range(1,T):
            for j in range(N):
                if i==j:
                    continue
                if (j in np.nonzero(h_plus_mask[k-1,i])[0]):
                    #dLLi_dxki_dxkj
                    val = self.dJi_dxi_dxj(x[k-1],u[k,i],i,j) + mu[k-1,i,j] * self.dh_dxi_dxj(x[k-1,i], x[k-1,j])
                    mtx = submtx(k,i,j)
                    mtx[:,:] = val
                    mtx = submtx(k,j,i)
                    mtx[:,:] = val.T
                    #dLLi_dxkj_dxkj
                    mtx = submtx(k,j,j)
                    mtx[:,:] = self.dJi_dxj_dxj(x[k-1],u[k,i],i,j) + mu[k-1,i,j] * self.dh_dxj_dxj(x[k-1,i], x[k-1,j])
                else:
                    #dLLi_dxki_dxkj
                    val = self.dJi_dxi_dxj(x[k-1],u[k,i],i,j) + self.dBh_dxi_dxj(x[k-1,i], x[k-1,j])
                    mtx = submtx(k,i,j)
                    mtx[:,:] = val
                    mtx = submtx(k,j,i)
                    mtx[:,:] = val.T
                    #dLLi_dxkj_dxkj
                    mtx = submtx(k,j,j)
                    mtx[:,:] = self.dJi_dxj_dxj(x[k-1],u[k,i],i,j) + self.dBh_dxj_dxj(x[k-1,i], x[k-1,j])
        k = T
        for j in range(N):
            if i==j:
                continue
            if (j in np.nonzero(h_plus_mask[k-1,i])[0]):
                #dLLi_dxki_dxkj
                val = self.dJfi_dxi_dxj(x[k-1],i,j) + mu[k-1,i,j] * self.dh_dxi_dxj(x[k-1,i], x[k-1,j])
                mtx = submtx(k,i,j)
                mtx[:,:] = val
                mtx = submtx(k,j,i)
                mtx[:,:] = val.T
                #dLLi_dxkj_dxkj
                mtx = submtx(k,j,j)
                mtx[:,:] = self.dJfi_dxj_dxj(x[k-1],i,j) + mu[k-1,i,j] * self.dh_dxj_dxj(x[k-1,i], x[k-1,j])
            else:
                #dLLi_dxki_dxkj
                val = self.dJfi_dxi_dxj(x[k-1],i,j) + self.dBh_dxi_dxj(x[k-1,i], x[k-1,j])
                mtx = submtx(k,i,j)
                mtx[:,:] = val
                mtx = submtx(k,j,i)
                mtx[:,:] = val.T
                #dLLi_dxkj_dxkj
                mtx = submtx(k,j,j)
                mtx[:,:] = self.dJfi_dxj_dxj(x[k-1],i,j) + self.dBh_dxj_dxj(x[k-1,i], x[k-1,j])


        if (self.DEBUG):
            '''
            for k in range(1,T+1):
                for ii in range(N):
                    for j in range(N):
                        mtx_num = submtx_num(k,ii,j)
                        mtx = submtx(k,ii,j)
                        if(not np.linalg.norm(mtx-mtx_num)<1e-4):
                            self.print_debug(f'i = {i} ii={ii},j={j},k={k}')
                            #breakpoint()
            '''
            dLL_dxdx_num = jacobianNumerical(lambda xx:self.dLLi_dx(xx.reshape(x.shape),u,h_plus_mask,lamda,mu,i), x.flatten(),dim=dim_x)
            self.print_debug(f'dLL_dxdx err {np.linalg.norm(dLL_dxdx_num - dLL_dxdx)}')
            assert(np.linalg.norm(dLL_dxdx_num - dLL_dxdx)<1e-4)
        if (self.CPP_DEBUG):
            alt = self.cpp.dLLi_dxdx([xx for xx in x],[uu for uu in u],[hh for hh in h_plus_mask],[ll for ll in lamda],[mm for mm in mu],i)
            if (np.linalg.norm(alt-dLL_dxdx)>1e-4):
                breakpoint()
        return dLL_dxdx

    '''
    # this derivative is identically zero
    def dLLi_dudx(self,x,u,h_plus_mask,lamda,mu,i):
        T = self.T; N = self.N; n = self.n; m = self.m; dim_x = T*N*n; dim_u = T*N*m
        return jacobianNumerical(lambda xx:self.dLLi_du(xx.reshape(x.shape),u,h_plus_mask,lamda,mu,i), x.flatten(),dim=dim_u)
    '''

    def dF_dx(self,x,u,i,k):
        if (self.USE_CPP):
            return self.cpp.dF_dx([xx for xx in x],[uu for uu in u],i,k)
        ''' F(x,u) = f(x_k_i,u_k_i)-x_k+1_i, find dF_dx, note x here is of dim(T*N*n) '''
        T = self.T; N = self.N; n = self.n; m = self.m; dim_x = T*N*n
        dFdx = np.zeros((n,dim_x))
        dFdx[:,(k-1)*N*n+i*n:(k-1)*N*n+(i+1)*n] = self.df_dx(x[k-1,i],u[k,i],i)
        dFdx[:,k*N*n+i*n:k*N*n+(i+1)*n] = -np.eye(n)
        if (self.CPP_DEBUG):
            alt = self.cpp.dF_dx([xx for xx in x],[uu for uu in u],i,k)
            if (np.linalg.norm(alt-dFdx)>1e-4):
                breakpoint()
        return dFdx

    def dF0_dx(self,x,u,i):
        if (self.USE_CPP):
            return self.cpp.dF0_dx([xx for xx in x],[uu for uu in u],i)
        ''' F0(x,u) = f(x_0_i,u_0_i)-x_1_i, find dF_dx note x here is of dim(T*N*n)
            A specialization for dF_dx when k=0, since we need x0
        '''
        T = self.T; N = self.N; n = self.n; m = self.m; dim_x = T*N*n
        dFdx = np.zeros((n,dim_x))
        dFdx[:,i*n:(i+1)*n] = -np.eye(n)
        if (self.CPP_DEBUG):
            alt = self.cpp.dF0_dx([xx for xx in x],[uu for uu in u],i)
            if (np.linalg.norm(alt-dFdx)>1e-4):
                breakpoint()
        return dFdx

    def dh_dx(self,x,k,i,j):
        if (self.USE_CPP):
            return self.cpp.dh_dx([xx for xx in x],k,i,j)
        ''' find d h(x_i,x_j)/ d x note x here is of dim(T*N*n) '''
        T = self.T; N = self.N; n = self.n; m = self.m
        dim_x = T*N*n
        dhdx = np.zeros((1,dim_x))
        dhdx[:,(k-1)*N*n+i*n:(k-1)*N*n+(i+1)*n] = self.dh_dxi(x[k-1,i],x[k-1,j])
        dhdx[:,(k-1)*N*n+j*n:(k-1)*N*n+(j+1)*n] = self.dh_dxj(x[k-1,i],x[k-1,j])
        if (self.CPP_DEBUG):
            alt = self.cpp.dh_dx([xx for xx in x],k,i,j)
            if (np.linalg.norm(alt-dhdx)>1e-4):
                breakpoint()
        return dhdx

    def dr_dx(self, x, u, lamda, mu, h_plus_mask):
        if (self.USE_CPP):
            return self.cpp.dr_dx([xx for xx in x],[uu for uu in u],[ll for ll in lamda],[mmm for mmm in mu],[hh for hh in h_plus_mask])
        ''' return: dim(r)*dim(x) '''
        T = self.T; N = self.N; n = self.n; m = self.m
        dim_x = T*N*n; dim_u = N*T*m
        dim_r = N*(T*n+T*m+T*n)+np.sum(h_plus_mask)
        drdx = np.zeros((dim_r,dim_x))
        index = 0
        for i in range(self.N):
            dLL_dxi_dx = self.dLLi_dxi_dx(x,u,h_plus_mask,lamda,mu,i)
            # this item is identically zero
            #dLL_dudx = np.zeros((dim_u,dim_x))
            dF0dx = self.dF0_dx(x,u,i)
            # dynamics for f(x0,u0) = x1
            drdx[index:index+T*n,:] = dLL_dxi_dx
            index += T*n + T*m
            drdx[index:index+n,:] = self.dynamics_residual_weight * dF0dx
            for k in range(1,self.T):
                dFdx = self.dF_dx(x,u,i,k)
                drdx[index+k*n:index+(k+1)*n,:] = self.dynamics_residual_weight * dFdx
            index += n*T
            for k in range(1,self.T+1):
                indices = np.nonzero(h_plus_mask[k-1,i])[0]
                if (len(indices) == 0):
                    continue
                dhdx = np.vstack([ self.dh_dx(x,k,i,j.item()) for j in indices ])
                drdx[index:index+dhdx.shape[0],:] = dhdx
                index += len(indices)

        if (self.DEBUG):
            drdx_num = jacobianNumerical(lambda xx:self.r(xx.reshape(x.shape),u,lamda,mu,h_plus_mask), x.flatten(),dim=dim_r)
            self.print_debug(f'drdx err {np.linalg.norm(drdx-drdx_num)}')
            assert(np.linalg.norm(drdx-drdx_num)<1e-4)
        if (self.CPP_DEBUG):
            alt = self.cpp.dr_dx([xx for xx in x],[uu for uu in u],[ll for ll in lamda],[mmm for mmm in mu],[hh for hh in h_plus_mask])
            if (np.linalg.norm(alt-drdx)>1e-4):
                breakpoint()
        return drdx

    '''
    def dr_dx_old(self, x, u, lamda, mu, h_plus_mask):
        # return: dim(r)*dim(x)
        T = self.T; N = self.N; n = self.n; m = self.m
        dim_x = N*T*n; dim_u = N*T*m
        dim_r = N*(dim_x+dim_u+T*n)+np.sum(h_plus_mask)
        #TODO change to fill-in style
        drdx = np.zeros((0,dim_x))
        for i in range(self.N):
            dLL_dxdx = self.dLLi_dxdx(x,u,h_plus_mask,lamda,mu,i)
            # this item is identically zero
            dLL_dudx = np.zeros((dim_u,dim_x))
            dF0dx = self.dF0_dx(x,u,i)
            # dynamics for f(x0,u0) = x1
            drdx = np.vstack([drdx,dLL_dxdx, dLL_dudx, dF0dx])
            for k in range(1,self.T):
                dFdx = self.dF_dx(x,u,i,k)
                drdx = np.vstack([drdx,dFdx])
            for k in range(1,self.T):
                drdx = np.vstack([drdx]+[ self.dh_dx(x,k,i,j.item()) for j in np.nonzero(h_plus_mask[k-1,i])[0] ])
            # h(x_T_i, x_T_j)
            drdx = np.vstack([drdx]+[ self.dh_dx(x,T,i,j.item()) for j in np.nonzero(h_plus_mask[T-1,i])[0] ])

        if (DEBUG):
            drdx_num = jacobianNumerical(lambda xx:self.r(xx.reshape(x.shape),u,lamda,mu,h_plus_mask), x.flatten(),dim=dim_r)
            self.print_debug(f'drdx err {np.linalg.norm(drdx-drdx_num)}')
            assert(np.linalg.norm(drdx-drdx_num)<1e-4)
        return drdx
    '''

    def dr_du(self, x, u, lamda, mu, h_plus_mask):
        if (self.USE_CPP):
            return self.cpp.dr_du([xx for xx in x],[uu for uu in u],[ll for ll in lamda],[mmm for mmm in mu],[hh for hh in h_plus_mask])
        ''' return: dim(r)*dim(u) '''
        T = self.T; N = self.N; n = self.n; m = self.m
        dim_x = T*N*n; dim_u = T*N*m
        dim_r = N*(T*n+T*m+T*n)+np.sum(h_plus_mask)


        drdu = np.zeros((dim_r,dim_u))
        index = 0
        for i in range(self.N):
            index += T*n
            for k in range(self.T):
                dLL_duik_duik = self.dJi_dudu(x[k-1],u[k,i],i)
                drdu[index+k*m:index+(k+1)*m, k*N*m+i*m:k*N*m+(i+1)*m] = dLL_duik_duik
            index += T*m
            k = 0
            drdu[index+k*n:index+(k+1)*n, k*N*m+i*m:k*N*m+(i+1)*m] = self.dynamics_residual_weight * self.df_du(self.x0[i],u[k,i],i)
            for k in range(1,self.T):
                drdu[index+k*n:index+(k+1)*n, k*N*m+i*m:k*N*m+(i+1)*m] = self.dynamics_residual_weight * self.df_du(x[k-1,i],u[k,i],i)
            index += n*T + np.sum(h_plus_mask[:,i]) # skip  f(x,u)-x+,  h(x,x)

        if (self.DEBUG):
            drdu_num = jacobianNumerical(lambda uu:self.r(x,uu.reshape(u.shape),lamda,mu,h_plus_mask), u.flatten(),dim=dim_r)
            if (np.linalg.norm(drdu-drdu_num)>1e-4):
                for i in range(N):
                    for k in range(0,T):
                        val = drdu[:,k*N*m + i*m:k*N*m+i*m+m]
                        val_num = drdu_num[:,k*N*m + i*m:k*N*m+i*m+m]
                        if (np.linalg.norm(val - val_num)>1e-4):
                            self.print_debug(f'k={k}, i={i},{np.nonzero(val-val_num)}')
                breakpoint()
            assert(np.linalg.norm(drdu-drdu_num)<1e-4)
        if (self.CPP_DEBUG):
            alt = self.cpp.dr_du([xx for xx in x],[uu for uu in u],[ll for ll in lamda],[mmm for mmm in mu],[hh for hh in h_plus_mask])
            if (np.linalg.norm(alt-drdu)>1e-4):
                breakpoint()
        return drdu

    def dr_dlamda(self, x, u, lamda, mu, h_plus_mask):
        if (self.USE_CPP):
            return self.cpp.dr_dlamda([xx for xx in x],[uu for uu in u],[ll for ll in lamda],[mmm for mmm in mu],[hh for hh in h_plus_mask])
        ''' return: dim(r)*dim(lamda) '''
        T = self.T; N = self.N; n = self.n; m = self.m
        dim_x = T*N*n; dim_u = T*N*m ; dim_lamda = T*N*n
        dim_r = N*(T*n+T*m+T*n)+np.sum(h_plus_mask)
        dr_dlamda = np.zeros((dim_r,dim_lamda))
        index = 0
        for i in range(N):
            for k in range(1,T):
                # dLLi_dxki_dlamda_ki
                dr_dlamda[index+(k-1)*n:index+k*n,k*N*n+i*n:k*N*n+(i+1)*n] = self.df_dx(x[k-1,i],u[k,i],i).T
                # dLLi_dxki_dlamda_k-1,i
                dr_dlamda[index+(k-1)*n:index+k*n,(k-1)*N*n+i*n:(k-1)*N*n+(i+1)*n] = -np.eye(n)
            k = T
            dr_dlamda[index+(k-1)*n:index+k*n,(k-1)*N*n+i*n:(k-1)*N*n+(i+1)*n] = -np.eye(n)
            index += T*n # skip dLL_dxi, index now points at dLLi_dui
            k = 0
            dr_dlamda[index+k*m:index+(k+1)*m,k*N*n+i*n:k*N*n+(i+1)*n] = self.df_du(self.x0[i],u[k,i],i).T
            for k in range(1,T):
                dr_dlamda[index+k*m:index+(k+1)*m,k*N*n+i*n:k*N*n+(i+1)*n] = self.df_du(x[k-1,i],u[k,i],i).T

            index += T*m + n*T + np.sum(h_plus_mask[:,i]) # skip  dLL_dui, f(x,u)-x+,  h(x,x)

        if (self.DEBUG):
            dr_dlamda_num = jacobianNumerical(lambda ll:self.r(x,u,ll.reshape(lamda.shape),mu,h_plus_mask), lamda.flatten(),dim=dim_r)
            self.print_debug(f'dr_dlamda err {np.linalg.norm(dr_dlamda-dr_dlamda_num)}')
            diff = dr_dlamda_num - dr_dlamda
            assert(np.linalg.norm(dr_dlamda-dr_dlamda_num)<1e-4)
        if (self.CPP_DEBUG):
            alt = self.cpp.dr_dlamda([xx for xx in x],[uu for uu in u],[ll for ll in lamda],[mmm for mmm in mu],[hh for hh in h_plus_mask])
            if (np.linalg.norm(alt-dr_dlamda)>1e-4):
                breakpoint()
        return dr_dlamda


    def dr_dmu(self, x, u, lamda, mu, h_plus_mask):
        if (self.USE_CPP):
            return self.cpp.dr_dmu([xx for xx in x],[uu for uu in u],[ll for ll in lamda],[mmm for mmm in mu],[hh for hh in h_plus_mask])
        ''' return: dim(r)*dim(mu) '''
        T = self.T; N = self.N; n = self.n; m = self.m
        dim_x = T*N*n; dim_u = T*N*m
        dim_r = N*(T*n+T*m+T*n)+np.sum(h_plus_mask)
        dim_mu = T*N*N

        dr_dmu = np.zeros((dim_r,dim_mu))
        index = 0
        for i in range(self.N):
            # dmu i,j,k
            dLL_dxi_dmu = self.dLLi_dxi_dmu(x,u,h_plus_mask,lamda,mu,i)
            dr_dmu[index:index+T*n,:] = dLL_dxi_dmu
            if (self.DEBUG):
                dLL_dxi_dmu_num = jacobianNumerical(lambda mm:self.dLLi_dxi(x,u,h_plus_mask,lamda,mm.reshape(mu.shape),i), mu.flatten(),dim=dim_x)
                assert(np.linalg.norm(dLL_dxi_dmu-dLL_dxi_dmu_num)<1e-4)

            index += T*n + T*m + n*T + np.sum(h_plus_mask[:,i])

        if (self.DEBUG):
            dr_dmu_num = jacobianNumerical(lambda mm:self.r(x,u,lamda,mm.reshape(mu.shape),h_plus_mask), mu.flatten(),dim=dim_r)
            self.print_debug(f'drdx err {np.linalg.norm(dr_dmu-dr_dmu_num)}')
            assert(np.linalg.norm(dr_dmu-dr_dmu_num)<1e-4)
        if (self.CPP_DEBUG):
            alt = self.cpp.dr_dmu([xx for xx in x],[uu for uu in u],[ll for ll in lamda],[mmm for mmm in mu],[hh for hh in h_plus_mask])
            if (np.linalg.norm(alt-dr_dmu)>1e-4):
                breakpoint()
        return dr_dmu


    # ---------- Defaults for  some Application specific functions -------
    # terminal(final) cost for agent i
    # x_T: terminal GAME state (N*n)
    # return : scalar
    # if User doesn't choose a terminal cost, the step cost J will be used
    def Jfi(self, x_T, i):
        return self.J(x_T,np.zeros(self.m),i)
    def dJfi_dxi(self, x_T, i):
        return self.dJi_dxi(x_T,np.zeros(self.m),i)
    def dJfi_dxj(self, x_T, i,j):
        return self.dJi_dxj(x_T,np.zeros(self.m),i,j)
    def dJfi_dxi_dxi(self, x_T, i):
        return self.dJi_dxi_dxi(x_T,np.zeros(self.m),i)
    def dJfi_dxi_dxj(self, x_T, i,j):
        return self.dJi_dxi_dxj(x_T,np.zeros(self.m),i,j)
    def dJfi_dxj_dxj(self, x_T, i,j):
        return self.dJi_dxj_dxj(x_T,np.zeros(self.m),i,j)

    # step cost function
    @abstractmethod
    def J(self,x_k,u_k_i,i):
        return 0
    @abstractmethod
    def dJi_dxi(self,x_k,u_k_i,i):
        return np.zeros((1,self.n))
    @abstractmethod
    def dJi_dxj(self,x_k,u_k_i,i,j):
        return np.zeros((1,self.n))
    @abstractmethod
    def dJi_dxi_dxi(self,x_k,u_k_i,i):
        return np.zeros((self.n,self.n))
    @abstractmethod
    def dJi_dxi_dxj(self,x_k,u_k_i,i,j):
        return np.zeros((self.n,self.n))
    @abstractmethod
    def dJi_dxj_dxj(self,x_k,u_k_i,i,j):
        return np.zeros((self.n,self.n))
    @abstractmethod
    def dJi_du(self,x_k,u_k_i,i):
        return np.zeros((1,self.m))
    @abstractmethod
    def dJi_dudu(self, x_k, u_k_i, i):
        return np.zeros((self.m,self.m))

