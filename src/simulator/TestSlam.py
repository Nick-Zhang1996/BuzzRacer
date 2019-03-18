#!/usr/bin/python

# EKF SLAM 
## Nick Zhang (nickzhang@gatech.edu)
import cv2
from math import sin, cos, radians, atan2 
from DataAssociator import DataAssociator
from time import time
import numpy as np
import matplotlib.pyplot as plt

class TestSlam:
    def __init__(self, simulation=True, simulator=None,verbose=True):
        self.verbose = verbose
        self.simulator = simulator
        self.simulation = True
        self.da = DataAssociator(simulation=simulation)
        np.set_printoptions(precision=2,linewidth=120)
        if simulation:
            self.stddev_observation = self.simulator.err_observation
            pass

        return


    def init(self, init_type = "Simulation", X = None, P = None):
        if init_type == "Simulation":
            self.position_covariance = []
            self.position_actual_err = []
            # [x,y,theta, x1, y1, ...].T n*1 np matrix
            if X is None:
                self.X = np.mat([0,0,0]).T
            else:
                self.X = X
            # n*n np matrix
            if P is None:
                self.P = np.mat(np.zeros((3,3)))
            else:
                self.P = P
            # observation, [x1,y1,x2,y2,...], 1*2n np array
            self.Z = None
            # format [[id_Z, id_X], [id_Z, id_X], ...], list of tuple or list
            # for new feature, id_X = -1
            self.association = None
            # list of coordinates of newly observed potential landmarks [[x,y],...]
            self.candidate_landmarks = []
            # threshhold distance for adding new landmark, landmark estimated to be in within this distance with an existing landmark will not be added. This should be cross referenced with parameters in data associator
            self.new_landmark_threshhold_distance = 1.0
            # respective seen and total count(two lists need to be updated simultaneously) 
            # We keed track of both because promotion of a candidate landmark is not a synchronized process
            self.seen_count = []
            self.total_count = []

            self.dx_err = 0.1*2
            self.dy_err = 0.1*2
            self.d_theta_err = radians(3.0)

            # limit on how much one update can affect state vector X (for x,y only)
            # currently there's no limit on the adjustment for robot orientation
            self.adjustment_limit = 5.0
            # DEBUG data
            self.n_updates = 0
            self.estimated_stddev_landmark = []
            self.estimated_stddev_location = []
            self.real_err = []

# make a prediction, dx, dy are translation of the robot since last call of predict()
# and are in world/map frame, in meters
# d theta in radians
    def predict(self, dx, dy, d_theta):
        self.X[0,0] += dx
        self.X[1,0] += dy
        self.X[2,0] += d_theta
        if (self.simulation):
            self.P[0,0] += (self.simulator.err_translation_update)**2
            self.P[1,1] += (self.simulator.err_translation_update)**2
            self.P[2,2] += (self.simulator.err_rotation_update)**2
        else:
            self.P[0,0] += (self.dx_err)**2
            self.P[1,1] += (self.dy_err)**2
            self.P[2,2] += (self.d_theta_err)**2

        return

# process a new observation
# NOTE: is this the best form?
# feature_vec: [x1,y1,x2,y2,....], in robot frame, meters
    def new_observation(self,feature_vec):
        if (feature_vec is None):
            return
        n_feature = len(feature_vec)/2
        n_confirmed_landmarks = (len(self.X)-3)/2
        # convert to map frame
        R = self.get_rotation_matrix(self.X[2,0])
        #R = self.get_rotation_matrix(self.simulator.robot_theta)
        feature_mat = np.empty([2,n_feature])
        feature_mat[0,:] = feature_vec[0::2]
        feature_mat[1,:] = feature_vec[1::2]
        feature_mat = np.mat(feature_mat)
        feature_mat = R*feature_mat + self.X[0:2,0]
        # Z for Data Association : [x1,y1,x2,y2,....], in map frame, meters
        Z = np.empty(2*n_feature)
        Z[0::2] = feature_mat[0,:]
        Z[1::2] = feature_mat[1,:]
        #print("Z= "+ str(Z))
        self.Z = Z

        self.association = self.da.associate(self.X, Z)

        if self.association is None :
            if self.verbose:
                print("no good feature")
            return

        else:
            # update
            # y = z-Hx, reorder z so H is I identity matrix when x is properly sliced
            associated = np.array(filter(lambda x:x[1]!=-1 and x[1]<n_confirmed_landmarks, self.association))
            if len(associated)<4:
                if self.verbose:
                    print("Not enough Associations")
            else:
                order = np.argsort(associated[:,1])
                # shape n*2
                sorted_associated = np.array([associated[i] for i in order])
                n_associated_feature = len(sorted_associated)
                # landmark index
                index = sorted_associated[:,1]
                # direct array index for X and P
                x_index = 2*index+3
                x_index = np.vstack([x_index,x_index+1])
                x_index = x_index.T.flatten()
                x_index = np.hstack([np.array([0,1,2]), x_index])
                # mesh grid
                rows = np.mat([x_index]*(n_associated_feature*2+3)).T
                cols = np.mat([x_index]*(n_associated_feature*2+3))

                # X and P use same index here
                reduced_X = self.X[x_index,:]
                reduced_P = self.P[rows,cols]
                assert(reduced_P.shape[0]==n_associated_feature*2+3)
                assert(reduced_P.shape[1]==n_associated_feature*2+3)

        # convert feature_vec to new_Z = [r_1, theta_1, r_2, theta_2,...]
        # because Jacobian for this form of new_Z is easier than Z=[x1,y1,x2,y2,...]
                r = [np.sqrt(feature_vec[2*i]**2+feature_vec[2*i+1]**2) for i in sorted_associated[:,0]]
                theta = [atan2(feature_vec[2*i+1], feature_vec[2*i]) for i in sorted_associated[:,0]]
                new_Z = np.mat(np.vstack([np.array(r),np.array(theta)]))
                new_Z = new_Z.T.flatten().T

                dx_map = [Z[2*i]-self.X[0,0] for i in sorted_associated[:,0]]
                dy_map = [Z[2*i+1]-self.X[1,0] for i in sorted_associated[:,0]]

                # old, dx dy Z
                #new_Z = [[Z[2*i],Z[2*i+1]] for i in sorted_associated[:,0]]
                #new_Z = np.mat(np.array(new_Z).flatten()).T
                #assert(len(new_Z)==2*n_associated_feature)

                # construct H, jacobian of expected observation
                # H = d(expected_observation(state))/d(state)
                # d/dx, to be precise it's partial differentiation
                # [d_r/dx_robot, d_theta/dx_robot]
                # NOTE possible optimization in all this
                c1 = [ [-dx_map[i]/r[i], dy_map[i]/(r[i]**2)] for i in range(n_associated_feature)]
                c1 = np.mat(np.array(c1).flatten()).T
                # [d_r/dy_robot, d_theta/dy_robot]
                c2 = [ [-dy_map[i]/r[i], -dx_map[i]/(r[i]**2)] for i in range(n_associated_feature)]
                c2 = np.mat(np.array(c2).flatten()).T
                # [d_r/d_theta_robot, d_theta/d_theta_robot]
                c3 = np.mat([0,-1]*n_associated_feature).T

                # [d_r_i/d_x_i, d_theta_i/d_y_i, .... ]
                # all this to avoid loops... just filling out a sparse matrix
                # cross derivative of two different landmark is zero
                c4_major = np.mat(np.diag( np.array([[dx_map[i]/r[i], dx_map[i]/(r[i]**2)] for i in range(n_associated_feature)]).flatten()))
                c4_minor_upper = np.mat(np.diag( np.array([[dy_map[i]/r[i], 0] for i in range(n_associated_feature)]).flatten()[:-1]))
                c4_minor_lower = np.mat(np.diag( np.array([[-dy_map[i]/(r[i]**2), 0] for i in range(n_associated_feature)]).flatten()[:-1]))
                c4_major[:-1,1:] += c4_minor_upper
                c4_major[1:,:-1] += c4_minor_lower
                c4 = c4_major

                H = np.hstack([c1,c2,c3,c4])

                # S=R + HPH.T (covariance matrix for innovation)
                # NOTE should we do x and y separtely?
                R = np.diag(np.ones(n_associated_feature*2)*(self.stddev_observation**2))
                S = H*reduced_P*H.T + R
                # K(gain) = P * H.T * inv(S)
                K = reduced_P * H.T * S.I

                # innovation y = Z-h(X)
                # Z_expected = [r;theta] = [sqrt((x-xr)^2-(y-yr)^2); atan2(x-xr,y-yr)-theta]
                x_r = self.X[0,0]
                y_r = self.X[1,0]
                theta_r = self.X[2,0]
                expected_Z = np.mat(np.hstack([np.sqrt(np.square(reduced_X[3::2]-x_r) + np.square(reduced_X[4::2]-y_r)), np.arctan2(reduced_X[4::2]-y_r,reduced_X[3::2]-x_r)-theta_r]).flatten()).T
                y = new_Z - expected_Z
                # wrap around angles, map everything to [-pi, pi]
                y[1::2] = (y[1::2] + np.pi) % (2*np.pi) - np.pi
                # apply a threshold for max adjustment
                adjustment_X = K*y
                adjustment_P = K*H*reduced_P
                mask = np.abs(adjustment_X)>self.adjustment_limit

                # if there are bad innovations
                # NOTE should we drop the whole thing?
                # Yes, it seems
                if np.any(mask):
                    return
                '''
                if np.any(mask):
                    adjustment_X[mask] = 0
                    bad_index = np.array(np.where(mask))
                    # reflect the selective blocking of large innovation here ( I can't think of a better way)
                    #for i in bad_index:
                    #    reduced_P[:,i] = 0
                    #    reduced_P[i,:] = 0

                '''
                # X = X + Ky
                reduced_X += adjustment_X

                # for optimal Kalman gain
                # P = (I-KH)P
                reduced_P -= adjustment_P

                prior_err = np.sqrt((self.X[0,0]-self.simulator.robot_x)**2+(self.X[1,0]-self.simulator.robot_y)**2)

                #print("heading correction: " +  str(reduced_X[2,0]-self.X[2,0]))
                self.X[x_index,:] = reduced_X
                self.P[rows,cols] = reduced_P
                #print("X = " + str(self.X))
                #print("P = " + str(self.P))
                post_err = np.sqrt((self.X[0,0]-self.simulator.robot_x)**2+(self.X[1,0]-self.simulator.robot_y)**2)
                if self.verbose:
                    self.position_actual_err.append(post_err)
                    #print("err = " +str(post_err))
                    pass

                #print("reduced err: " + str(prior_err-post_err))
                #print("heading diff: " + str(self.X[2,0]-self.simulator.robot_theta))
                average = np.sqrt(np.average([self.P[i,i] for i in range(self.P.shape[0])]))
                stddev_loc = np.sqrt((self.P[0,0]+self.P[1,1])/2)
                if self.verbose:
                    self.position_covariance.append(average)
                    if (average<0.95*post_err):
                        print("err = " +str(post_err))
                        print("estimated err = " + str(average))

                self.n_updates += 1
                self.estimated_stddev_landmark.append(average)
                self.real_err.append(post_err)
                self.estimated_stddev_location.append(stddev_loc)
                #NOTE tunable parameter
                if post_err>3.0:
                    raise Exception

# --------------- new landmarks -------------

            temp_landmarks = np.mat(self.candidate_landmarks).flatten().T
            self.association = self.da.associate(np.vstack([self.X,temp_landmarks]), Z)
            self.total_count = [x+1 for x in self.total_count]
            # track new landmarks
            new_landmarks = filter(lambda x:x[1]==-1, self.association)
            n_new_landmarks = len(new_landmarks)
            # check new landmarks afterwards DEBUG
            for hypothesis in new_landmarks:
                id_Z = hypothesis[0]
                self.candidate_landmarks.append([Z[2*id_Z],Z[2*id_Z+1]]) 

            self.seen_count.extend([1]*n_new_landmarks)
            self.total_count.extend([0]*n_new_landmarks)

            # update seen and unseen count for candidate landmarks
            # [id_Z, id_X]
            reseen_landmarks = filter(lambda x:x[1]!=-1 and x[1]>=n_confirmed_landmarks, self.association)
            for i in range(len(reseen_landmarks)):
                id_Z = reseen_landmarks[i][0]
                # id_X - n_confirmed_landmarks = id_candidate_landmarks
                id_cl = reseen_landmarks[i][1]-n_confirmed_landmarks
                x,y = self.candidate_landmarks[id_cl]
                x = (x*self.seen_count[id_cl] + Z[id_Z*2])/(self.seen_count[id_cl]+1)
                y = (y*self.seen_count[id_cl] + Z[id_Z*2+1])/(self.seen_count[id_cl]+1)
                self.candidate_landmarks[id_cl] = [x,y]
                self.seen_count[id_cl] += 1

            #print(self.candidate_landmarks)
            #print(self.seen_count)
            #print(self.total_count)

            # promote qualified candidate landmarks (NOTE this can be done at a lower frequency if needed)
            # and delete unpromising candidates
            # NOTE, tunable parameters here
            i=0
            while (i<len(self.candidate_landmarks)):
                if self.total_count[i] > 20:
                    if self.seen_count[i]< 0.8 * self.total_count[i] :
                        del self.candidate_landmarks[i]
                        del self.seen_count[i]
                        del self.total_count[i]
                    else:
                        # make sure we don't re-add an existing landmark
                        # NOTE tunable parameter here
                        flag_bad_landmark = False
                        for id_X in range((self.X.shape[0]-3)/2):
                            if np.sqrt(np.square(self.X[2*id_X+3,0]-self.candidate_landmarks[i][0])+np.square(self.X[2*id_X+4,0]-self.candidate_landmarks[i][1])) < self.new_landmark_threshhold_distance**2 :
                                flag_bad_landmark = True
                                break
                        if not flag_bad_landmark:
                            pass
                            self.add_landmark(self.candidate_landmarks[i])
                        del self.candidate_landmarks[i]
                        del self.seen_count[i]
                        del self.total_count[i]
                i+=1
        return

# add a landmark to state vector self.X
    def add_landmark(self, coordinate):
        new_X = np.empty((self.X.shape[0]+2,self.X.shape[1]))
        new_X[:-2, :] = self.X
        new_X[-2] = coordinate[0]
        new_X[-1] = coordinate[1]
        self.X = np.mat(new_X)

        dist_squared = np.square(coordinate[0]-self.X[0,0]) + np.square(coordinate[1]-self.X[1,0])

        new_P = np.zeros((self.P.shape[0]+2,self.P.shape[1]+2))
        new_P[:-2, :-2] = self.P
        new_P[-2,-2] = self.stddev_observation**2 + self.P[0,0] + self.P[2,2]*dist_squared
        new_P[-1,-1] = self.stddev_observation**2 + self.P[1,1] + self.P[2,2]*dist_squared
        self.P = np.mat(new_P)

    def get_rotation_matrix(self,rad):
# generate a 2d rotation matrix x'=Rx, thera in radians
        R = [[cos(rad), -sin(rad)], [sin(rad), cos(rad)]]
        return np.array(R)
# show some trends of recorded data
    def debug_trend(self):
        #t_range = np.linspace(0.0,t_final, int(t_final/t_stepsize)+1)
        t_range = np.linspace(0.0,self.simulator.t_end, self.n_updates)
        #fix,ax = plt.subplots(2,2)

        #ax[0,0].plot(t_range, self.estimated_stddev_location, "r-", t_range, self.real_err, "b-", t_range, self.estimated_stddev_landmark, "g--.")
        #ax[0,0].set_title("err location")
        plt.plot(t_range, self.estimated_stddev_location, "r-", t_range, self.real_err, "b-", t_range, self.estimated_stddev_landmark, "g--.")
        #plt.set_title("err location")

        #ax[0,1].plot(t_range, gain)
        #ax[0,1].set_title("gain")

        #ax[1,0].plot(t_range, variance_pos)
        #ax[1,0].set_title("variance position")
        ##ax[1,1].plot(t_range, estimated_speed,"b-",t_range, real_speed, "r--")
        ##ax[1,1].set_title("estimated_speed")
        #ax[1,1].plot(t_range, acc_history)
        #ax[1,1].set_title("acc_history")

        plt.show()
        pass
    def showErr(self):
        real = np.array(self.position_actual_err)
        estimation = np.array(self.position_covariance)

        plt.plot(real,label='real error')
        plt.plot(estimation,label='estimated error')
        plt.legend()
        plt.show()


