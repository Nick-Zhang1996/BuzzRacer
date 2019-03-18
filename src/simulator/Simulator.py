#!/usr/bin/python

## Simulator for testing SLAM, and Data Association
## Nick Zhang Fall 2018 (nickzhang@gatech.edu)
import cv2
import matplotlib.pyplot as plt
from math import sin, cos, radians, sqrt, atan2, ceil,floor
from DataAssociator import DataAssociator
from TestSlam import TestSlam
from time import time

import numpy as np

# This simulator contains two relatively independent components
# One for testing data association, in which a set of X and Z is generated, and client DA is expected to find the correct association between them.(i.e. the observation matrix H), multiple independent simiulations are executed, and statistics on false positives, false negatives are shown. It is also possible to visualize the associations to examine fail cases.

# The other is a discrete time, stepwise simulation of a complete slam implementation. Landmarks are randomly generated and remains constant throughout the simulation, and the robot moves passively with a predefined path. Features observed are pushed to client slam instance, who executes required calculations for state estimations. A real-time view is generated during the simulation.

# for any simulation to work, the client class to be tested need to possess some required APIs and member variables, see comments and examples for more information. You will need to read code closely to understand expected representation of data, it can be quite confusing. 

# Many functions are similar but tailored for different simulation scenarios, make sure you modify the correct one.

class Simulator :
    # Init, setup vars
    def __init__(self, verbose=False):
# --- Common Settings

        self.verbose = verbose
#  map size, in meters ( for SLAM this is whole map)
        self.map_size = 100
# pixels per meter
        self.resolution = 10.0
        self.association = None
# stddev of error in feature detection, note this is in x and y, discretely
        self.err_observation = 0.1
# possibility of random false positive features in every feature update
        self.features_false_positive = 0.1
        self.features_false_negative = 0.1

# --- Data Association related parameters ----

# num of total landmarks (DA specific)
        self.n_landmarks = 13
# error in robot pose (DA specific)
        self.err_robot = 5.0
# error in robot's orientation, in degrees, (DA specific)
        self.robot_angle_err = radians(7.0)

# --- SLAM related parameters

        self.features = None
# simulation update time interval(in sec), also time between SLAM predictions, or kinematics updates
        self.time_interval = 0.1
# number of sensor updates (slam updates) between each simulation update
# in reality, LIDAR runs at 25hz, while odom updates at 10Hz, so multiple feature updates are done between predictions
# in perfect simulation the observations are made with respect to different robot states, but here we assume robot doesn't move(in realityit moves very little)
        self.n_observations_between_predictions = 2
        # number of observations before the robot starts moving
        # NOTE this is essential to a successful simulation
        self.n_pre_observations = 20
# LIDAR range (meters, radius)
        self.lidar_range = 60.0
#  stddev of error of kinematics in each step, gaussian
        self.err_translation_update = 0.2
        self.err_rotation_update = radians(3.0)

# set the trajectory in self.slam_test_prep()
# m/s
        self.robot_mean_speed = 1.0

# stddev of movement for the robot between each updates, before error is added
        self.rotation_mean = 0.0
# unit: degrees
        self.rotation_stddev = radians(20.0)
        self.n_slam_false_positive = 0
        self.n_slam_landmarks = 15

    def test_slam(self):
# run a simulation for SLAM
# to cheat a bit, let robot stay stationary and observe landmark without moving 
        for i in range(self.n_pre_observations):
# format [x1,y1,x2,y2,...]
            feature_vector = self.make_observation()
            self.slam.new_observation(feature_vector)

        try:
            while (self.t < self.t_end):
                self.update_simulation()
                self.visualize_slam()
                if self.verbose:
                    print("Progress: " +str(self.t/self.t_end))
        except Exception:
            print("SLAM diverge, aborting simulation")
        finally:
            cv2.destroyAllWindows()

        return
    def update_boundary_from_X(self):
# update coordinates for canvas boundaries, include all landmarks
# specific for data associator
# return [[x,y],[x,y]], top left, bottom right coordinate in world frame, in meters, float 
# for visualize()
        max_x = np.max(self.X[3::2])
        max_y = np.max(self.X[4::2])
        min_x = np.min(self.X[3::2])
        min_y = np.min(self.X[4::2])

        max_x = np.max([max_x,self.X[0]]) + 10
        max_y = np.max([max_y,self.X[1]]) + 10
        min_x = np.min([min_x,self.X[0]]) - 10
        min_y = np.min([min_y,self.X[1]]) - 10

        self.boundary = np.array( [[min_x,min_y], [max_x,max_y]] )

        return self.boundary

    def update_boundary_from_landmarks(self):
# update coordinates for canvas boundaries, include all landmarks, robot position, etc
# this is not cumulative
# return [[x,y],[x,y]], top left, bottom right coordinate in world frame, in meters, float 
        max_x = np.max(self.landmarks[0,:]) 
        max_y = np.max(self.landmarks[1,:]) 
        min_x = np.min(self.landmarks[0,:]) 
        min_y = np.min(self.landmarks[1,:]) 

        max_x = np.max([max_x,self.robot_x]) + 10
        max_y = np.max([max_y,self.robot_y]) + 10
        min_x = np.min([min_x,self.robot_x]) - 10
        min_y = np.min([min_y,self.robot_y]) - 10

        self.boundary = np.array( [[min_x,min_y], [max_x,max_y]] )

        return self.boundary

    def update_boundary_from_robot_position(self):

        max_x = np.max([self.boundary[1,0]-10,self.robot_x]) + 10
        max_y = np.max([self.boundary[1,1]-10,self.robot_y]) + 10
        min_x = np.min([self.boundary[0,0]+10,self.robot_x]) - 10
        min_y = np.min([self.boundary[0,1]+10,self.robot_y]) - 10

        self.boundary = np.array( [[min_x,min_y], [max_x,max_y]] )
        return self.boundary

    def world_to_canvas(self,cord_world):
#conver world corrdinate (in meter) to canvas coordinate ( in pixels)
# cord_world : [x,y] (meter) 
# cord_canvas : [x,y]  (pixels) in int
        cord_world = np.squeeze(np.array(cord_world))
        offset = cord_world - self.boundary[0,:]
        cord_canvas = np.int_(offset*self.resolution)
        if not np.all(cord_canvas >= 0):
            if self.verbose:
                print("warning, marker outside of canvas. " +str(cord_world))
            return None
        #assert all(cord_canvas >= 0)
        return tuple(cord_canvas)

    def create_canvas(self):
# create a canvas from boundary, resolution
# call update_boundary_from_X() beforehand
        canvas_size = self.boundary[1,:]-self.boundary[0,:]
        canvas_size = canvas_size * self.resolution
        canvas_size = np.int_(canvas_size) + 1
        assert all(canvas_size>0)
        return np.ones(np.hstack([canvas_size[::-1],3]),dtype = np.uint8)*255

    def dummy_data_associator_test_prep(self):
# testing parameters
        self.X = [0,0,0, 10,10, 10,20, 20,10]
        self.X = np.array(self.X).T

        self.Z = [ 13,13, 11.1, 20.5, 20.1, 10.3]
        self.Z = np.array(self.Z).T

        self.H = np.vstack([np.zeros([3,6]), np.identity(6)])
        print("errors")
        print(self.X-np.matmul(self.H, self.Z))

        self.association = np.array([[0,0],[1,1],[2,2]],dtype = np.uint8)
        return

    def get_rotation_matrix(self,rad):
# generate a 2d rotation matrix x'=Rx, thera in radians
        R = [[cos(rad), -sin(rad)], [sin(rad), cos(rad)]]
        return np.array(R)

    def data_associator_test_prep(self):
# prepare a test for DA
# number of landmarks detected as features, the actual feature number is larger(false pos)_
        n_features = int(self.n_landmarks*(1-self.features_false_negative))
        n_fake_features = int(self.n_landmarks*(self.features_false_positive))

# generate X and Z randomly
        self.x_coord = np.random.random_integers(0,self.map_size, size = self.n_landmarks)
        self.y_coord = np.random.random_integers(0,self.map_size, size = self.n_landmarks)

        self.X = np.empty((self.n_landmarks*2+3,1))
# robot position
        self.X[0,0] = self.map_size/2
        self.X[1,0] = self.map_size/2
        self.X[2,0] = radians(30.0)
        self.X[3::2,0] = self.x_coord.T
        self.X[4::2,0] = self.y_coord.T

# convert landmarks to robot frame
        landmark_x = self.X[3::2,0].copy()
        landmark_y = self.X[4::2,0].copy()
        landmark_x -= self.X[0,0]
        landmark_y -= self.X[1,0]

# select a subset of landmarks to simulate failure in feature detection, shuffle order
        selected = np.arange(self.n_landmarks)
        np.random.shuffle(selected)
        selected = selected[:n_features]
        selected = list(selected)
        landmark_x = [landmark_x[i] for i in selected]
        landmark_y = [landmark_y[i] for i in selected]

# transform to robot frame
        coord = np.vstack([landmark_x,landmark_y])
        R = self.get_rotation_matrix(-self.X[2,0])
        coord = np.matmul(R, coord)

# if more needs to be done in robot frame, add here

# apply translation error, random offset to each feature
        coord = coord + np.random.normal(scale = self.err_observation, size = (2,n_features))

# apply angular error, and rotate back
        R = self.get_rotation_matrix(self.X[2,0]+self.robot_angle_err)
        coord = np.matmul(R, coord)
        landmark_x = coord[0,:]
        landmark_y = coord[1,:]
        landmark_x += self.X[0,0] + self.err_robot
        landmark_y += self.X[1,0] + self.err_robot
        self.Z = np.empty(2*len(landmark_x))
        self.Z[0::2] = landmark_x
        self.Z[1::2] = landmark_y
# add false positives
        fake_features = np.random.random_integers(0,self.map_size, size = 2*n_fake_features)
        self.Z = np.hstack([self.Z, fake_features])

        self.true_association = np.vstack([ range(0,len(landmark_x)), selected]).T

        return

    def make_observation(self):
        #BUG: when theta<0 can't detect feature properly
# generate a observation from self.landmarks, with respect to robot_x, robot_y, robot_theta
# this function simulates feature extraction process
# return: np array: [[x1,x2,x3,..],[y1,y2,y3,...]], with respect to robot

# number of landmarks detected as features, the actual feature number is larger(false pos)_
        n_fake_features = self.n_slam_false_positive

        possible_landmark = self.landmarks.copy()
        possible_landmark = list(possible_landmark.T)
        possible_landmark = filter(lambda entry: ((entry[0]-self.robot_x)**2+(entry[1]-self.robot_y)**2)<self.lidar_range**2 , possible_landmark)
        n_possible_landmark = len(possible_landmark)

# prepare false positives NOTE: this is not uniform distribution
# one hickup is that the fake ones are always at the end, this may have an effect on data associator
        theta = np.random.uniform(0,2*np.pi,size = n_fake_features)
        r = np.random.uniform(2,self.lidar_range ,size = n_fake_features)
        x = np.cos(theta)*r
        y = np.sin(theta)*r
        fake_features = np.vstack([x,y])

        if n_possible_landmark == 0:
            self.possible_landmark = None
            self.true_association = None
            self.features = fake_features
            return None

# select a subset of landmarks to simulate failure in feature detection, shuffle order
        selected = np.arange(n_possible_landmark)
        np.random.shuffle(selected)
        selected = selected[:int(ceil(n_possible_landmark*(1-self.features_false_negative)))]
        #print("DEBUG: omitting "+str(n_possible_landmark - len(selected)) + " landmarks")
        selected = list(selected)
        n_detected_landmark = len(selected)
        detected_landmark = [possible_landmark[i] for i in selected]
        detected_landmark = np.mat(detected_landmark).T

# transform to robot frame, features shape (2,n)
        detected_landmark -= np.mat([self.robot_x,self.robot_y]).T
        R = self.get_rotation_matrix(-self.robot_theta)
        features = np.matmul(R, detected_landmark)

# if more needs to be done in robot frame, add here

# apply translation error, random offset to each feature
# TODO, add noise at r,theta stage to confirm with slam noise model
        features = features + np.random.normal(scale = self.err_observation, size = (2,n_detected_landmark))


        # in robot frame, matrix(2*m)
        features = np.hstack([features, fake_features])
# for visualization from id in feature vector to id in possible landmark

        self.possible_landmark = possible_landmark
        self.true_association = np.vstack([range(0,n_detected_landmark), selected]).T
        observation = np.empty((2*(n_fake_features+len(selected))))
        observation[0::2] = features[0,:]
        observation[1::2] = features[1,:]

        features_in_world = features.copy()
# transform to world frame, with real robot state
        R = self.get_rotation_matrix(self.robot_theta)
        features_in_world = np.matmul(R, features_in_world)
        features_in_world[0,:] += self.robot_x
        features_in_world[1,:] += self.robot_y
        self.features = features_in_world

        features_in_estimated_world = features.copy()

# transform to world frame, with estimated robot state
        R = self.get_rotation_matrix(self.slam.X[2,0])
        features_in_estimated_world = np.matmul(R, features_in_estimated_world)
        features_in_estimated_world[0,:] += self.slam.X[0,0]
        features_in_estimated_world[1,:] += self.slam.X[1,0]
        self.features_by_robot = features_in_estimated_world

        return observation

    def init_da_statistics(self):
    # init and clear the counting variables for da statistics
        self.n_mismatch = 0
        self.n_notFound = 0
        self.n_total_feature = 0
        return

    def evaluate_da(self, client_association, visualize_at_err):
# compare client prediction with real association, update statistics
        self.association = client_association
# find number of mismatched (BAD, causes divergence)
# and number of ignorance/false negatives (Acceptable)
        self.n_total_feature += len(self.true_association)
        debug_old_mismatch = self.n_mismatch
        for pair in self.true_association:
            isFound = False
            isMismatch = False
            for guess in self.association:
                if (pair[0] == guess[0]):
                    if (pair[1] == guess[1]):
                        isFound = True
                    else:
                        isMismatch = True
                    break
            if not isFound:
                self.n_notFound += 1
            if isMismatch:
                self.n_mismatch += 1

        if (visualize_at_err and self.n_mismatch>debug_old_mismatch):
            self.visualize()
            pass
    def show_da_statistics(self, da):

        print("nn_distance:"+str(da.nn_distance))
        print("joint_distance:"+str(da.joint_distance))
        print("clear_distance" + str(da.clear_distance))
        print("Percentage Mismatch " + str(float(self.n_mismatch)*100/self.n_total_feature)+"%")
        print("Percentage not found " +  str(float(self.n_notFound)*100/self.n_total_feature)+"%")
        return

    def drawCross(self, canvas, coord, size = 7, thickness = 2, color = (25,25,25)):
# coord unit: pixels
        point1 = np.array([-size,-size]) + coord
        point2 = np.array([size,size]) + coord
        point1 = tuple(point1)
        point2 = tuple(point2)
        cv2.line(canvas, point1, point2, color, thickness)

        point1 = np.array([size,-size]) + coord
        point2 = np.array([-size,size]) + coord
        point1 = tuple(point1)
        point2 = tuple(point2)
        cv2.line(canvas, point1, point2, color, thickness)

    def slam_test_prep(self, slam = None):
# initialize a test for SLAM testing
# the slam instance given should have the following properties:
# slam.X
#       : state matrix, np.mat type, shape (2*n+3,1), [robot_x, robot_y, robot_theta, landmark_1_x, landmark_1_y, landmark_2,......].T
# slam.P
#       : Covariance matrix for state space, shape (2*n+3, 2*n+3)
# slam.init(type="Simulation", X, P)
#       : initialize with given X and P
# slam.predict(dx,dy,d_theta)
#       : make a prediction given dx,dy,d_theta, all in map ref frame, first a translation (dx,dy), then a rotation d_theta
# slam.association
#       : association from feature list to slam.X, use requence id not matrix index, list of tuples [(id_z, id_x),..]
# slam.new_observation(feature_vec)
#       : register a new list of features, frame: robot, format: [x1,y1,x2,y2,...]
# slam.update(H, Z) (unused in here)
#       : make a update given observation matrix H and Z, 
#       : H is the association matrix, shape (n_feature,2), each entry is [id_feature, id_landmark], in real order, not actual index (*2+3)
#   (Maybe we should update H so that X-HZ = y(innovation), but H may be very large that way, and we wouldn't be able to do cool submatrix operation)
#       : Z is the observation vector [x1,y1,x2,y2,...], this is in map/world frame

        self.slam = slam
# simulation time
        self.t = 0.0
        self.t_last = None
        self.t_end = 16.0
# define robot trajectory(so it doesn't goes off map)
# trajectory as y = f(x), map/world frame, in meters
        #self.traj = lambda x: 0.005*x**2
        self.traj = lambda x: 32+30*sin(2*np.pi/50*x)
# derivatives for calculating correct step size between each time interval
# df(x)/dx, as a function of x
        #self.traj_dydx = lambda x:0.01*x
        self.traj_dydx = lambda x:30*cos(2*np.pi/50*x)*2*np.pi/50
# dx/dt, given v(speed) and x
        self.traj_dxdt = lambda x,v: v*sqrt(1.0/(1.0+self.traj_dydx(x)**2))
        self.robot_x = 0.0
        self.robot_y = self.traj(self.robot_x)
        self.robot_theta = atan2(self.traj_dydx(self.robot_x), 1) 

        X = np.mat([self.robot_x, self.robot_y, self.robot_theta]).T
        P = np.mat(np.zeros((3,3)))
        # NOTE DEBUG:
        #X = np.mat([self.robot_x, self.robot_y, self.robot_theta,10,10,20,20,60,60]).T
        X = np.mat([self.robot_x, self.robot_y, self.robot_theta]).T
        #P = np.mat(np.diag(3*[0]+6*[self.err_observation]))
        P = np.mat(np.diag(3*[0]))
# create landmarks
#TODO make this float
        self.landmarks = np.random.random_integers(0,self.map_size, size = (2, self.n_slam_landmarks)).astype(np.float64)
        # DEBUG: static landmarks
        #self.landmarks = np.array([[10,10],[20,20],[20,40], [60,60]],dtype=np.float).T
# update boundaries
        self.update_boundary_from_landmarks()
        self.slam.init("Simulation", X, P)
        return

    def update_simulation(self):
# update world by self.time_interval, call slam.predict() and update()

        self.t += self.time_interval
# calc translation and rotation, in map/world frame
# convention: first a translation, then a rotation
# TODO, implement variation in speed
        dx = self.traj_dxdt(self.robot_x, self.robot_mean_speed)
        dy = self.traj_dydx(self.robot_x)*dx # or we can use a subtraction for this
        d_theta = atan2(self.traj_dydx(self.robot_x+dx), 1) - self.robot_theta

        self.robot_x += dx
        self.robot_y = self.traj(self.robot_x)
        #print("pos: "+str((self.robot_x, self.robot_y)))
        self.robot_theta = atan2(self.traj_dydx(self.robot_x), 1)

# To simulate the accumulative error resulted from a heading offest error, we convert this kinematics update to a different, intermediate form(dr d_theta). However, we convert this back to dx dy d_theta with estimated heading(which would contain accumulated error if slam isn't correcting heading properly) for the actual predict() because that's the representation for Husky robot (pose differentiation)
# in real life this conversion back to dx dy is done internally with robot odometry system

        # distance travelled along current heading
        dr = np.sqrt(dx**2+dy**2)
        
        noise_dr = np.random.normal(0.0, scale = self.err_translation_update)
        noise_d_theta = np.random.normal(0.0, scale = self.err_rotation_update)
        dr += noise_dr
        d_theta += noise_d_theta

        dx = dr * cos(self.slam.X[2,0])
        dy = dr * sin(self.slam.X[2,0])

# call slam prediction step
        self.slam.predict( dx, dy, d_theta)
# make several observations, update slam after each
        for i in range(self.n_observations_between_predictions):
# format [x1,y1,x2,y2,...]
            feature_vector = self.make_observation()
            self.slam.new_observation(feature_vector)
            #self.visualize_slam(static=True)
        return

    def visualize(self):
# visualize landmark, feature, and association
        self.update_boundary_from_X()
        canvas = self.create_canvas()

# draw X, landmarks in X will be black crosses
        for (x,y) in zip(self.X[3::2,0],self.X[4::2,0]):
            coord = self.world_to_canvas((x,y))
            self.drawCross(canvas, coord)

# draw Z, Z will be RED circles
        for (x,y) in zip(self.Z[::2],self.Z[1::2]):
            coord = self.world_to_canvas((x,y))
            cv2.circle(canvas, coord, radius = 4, color = (255,0,0), thickness = 2)

# draw robot, BLUE
        coord = self.world_to_canvas((self.X[0,0],self.X[1,0]))
        cv2.circle(canvas, coord, radius = 6, color = (0,0,255), thickness = 3)
        
# draw client association, with BLUE line, new landmarks are BIG RED cross
        for entry in self.association:
            id_z = entry[0]
            id_x = entry[1]
            if (id_x == -1):
                # new landmarks
                point = self.world_to_canvas(self.Z[id_z*2:id_z*2+2])
                self.drawCross(canvas, point, size = 7, thickness = 2, color = (255,0,0))

            else:
                point1 = self.world_to_canvas(self.X[3+id_x*2:3+id_x*2+2,0])
                point2 = self.world_to_canvas(self.Z[id_z*2:id_z*2+2])
                cv2.line(canvas, point1, point2, color = (0,0,255), thickness = 3)


# draw actual association, with GREEN line
        for entry in self.true_association:
            id_z = entry[0]
            id_x = entry[1]
            point1 = self.world_to_canvas(self.X[3+id_x*2:3+id_x*2+2,0])
            point2 = self.world_to_canvas(self.Z[id_z*2:id_z*2+2])
            cv2.line(canvas, point1, point2, color = (0,255,0), thickness = 2)

# draw a legend indicating scale
        y_size = canvas.shape[0]
# in meter
        legend_length = 1.0
        thickness = 1
        cv2.line(canvas,(10, y_size-30),(int(10+legend_length*self.resolution), y_size-30), (0,0,0), thickness) 
        cv2.line(canvas,(10, y_size-30),(10, y_size-30-6), (0,0,0), thickness) 
        cv2.line(canvas,(10+int(legend_length*self.resolution), y_size-30-6),(int(10+legend_length*self.resolution), y_size-30), (0,0,0), thickness) 
        text = "1m"
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(canvas, text, (10, y_size-14), font, 0.4, (0,0,0), thickness,cv2.LINE_AA)

        plt.imshow(canvas)
        plt.show()

        return

    def visualize_slam(self, static=False):
        self.update_boundary_from_robot_position()
# visualize robot, landmark, feature, association, and covariance
        canvas = self.create_canvas()

# draw true landmarks as black crosses
        for (x,y) in zip(np.squeeze(np.asarray(self.landmarks[0,:])),np.squeeze(np.asarray(self.landmarks[1,:]))):
            coord = self.world_to_canvas((x,y))
            if coord is not None:
                self.drawCross(canvas, coord)

# draw last updated features as RED circles, this is their actual location, not where the robot thinks they are
        if not self.features is None and not len(self.features)==0:
            #for (x,y) in zip(np.squeeze(np.asarray(self.features[0,:])),np.squeeze(np.asarray(self.features[1,:]))):
            for [x,y] in np.vstack([np.squeeze(np.asarray(self.features[0,:])),np.squeeze(np.asarray(self.features[1,:]))]).T:
                coord = self.world_to_canvas((x,y))
                if coord is not None:
                    cv2.circle(canvas, coord, radius = 4, color = (255,0,0), thickness = 2)
        else:
            print("no feature")

# draw robot, BLUE circle with a heading (real location)
        coord = self.world_to_canvas((self.robot_x, self.robot_y))
        cv2.circle(canvas, coord, radius = 16, color = (0,0,255), thickness = 3)
        dx = 3.0*cos(self.robot_theta)
        dy = 3.0*sin(self.robot_theta)
        coord2 = self.world_to_canvas((self.robot_x+dx, self.robot_y+dy))
        cv2.line(canvas, coord, coord2, (0,0,255), thickness = 2)

# draw robot, BLUE circle with a heading (estimated location)
        coord = self.world_to_canvas((self.slam.X[0,0], self.slam.X[1,0]))
        cv2.circle(canvas, coord, radius = 16, color = (100,100,255), thickness = 3)
        dx = 3.0*cos(self.slam.X[2,0])
        dy = 3.0*sin(self.slam.X[2,0])
        coord2 = self.world_to_canvas((self.slam.X[0,0]+dx, self.slam.X[1,0]+dy))
        cv2.line(canvas, coord, coord2, (100,100,255), thickness = 2)

# draw client association, with BLUE line, new landmarks are BIG RED cross
        if self.slam.association is None:
            print("No client association available")
        else:
            for entry in self.slam.association:
                id_z = entry[0]
                id_x = entry[1]
                if (id_x == -1):
                    point = self.world_to_canvas(self.features_by_robot[:,id_z])
                    if point is not None:
                        # new landmarks/ unassociated features
                        self.drawCross(canvas, point, size = 9, thickness = 2, color = (255,0,0))

                elif (id_x < (self.slam.X.shape[0]-3)/2):
                    # associated landmarks
                    # landmark location
                    point1 = self.world_to_canvas(self.slam.X[3+id_x*2:3+id_x*2+2,0])
                    # feature location
                    point2 = self.world_to_canvas(self.features_by_robot[:,id_z])
                    if point1 is not None and point2 is not None:
                        cv2.line(canvas, point1, point2, color = (0,0,255), thickness = 3)
                    else:
                        print("err: client association")


# draw actual association, with GREEN line (from landmark to robot's reconstruction of feature location)
        if self.true_association is None:
            print("no true association")
        else:
            for entry in self.true_association:
                id_z = entry[0]
                id_x = entry[1]
                point1 = self.world_to_canvas(self.possible_landmark[id_x])
                point2 = self.world_to_canvas(np.squeeze(self.features_by_robot[:,id_z]))
                cv2.line(canvas, point1, point2, color = (0,255,0), thickness = 2)

# draw a legend indicating scale
        y_size = canvas.shape[0]
# in meter
        legend_length = 1.0
        thickness = 1
        cv2.line(canvas,(10, y_size-30),(int(10+legend_length*self.resolution), y_size-30), (0,0,0), thickness) 
        cv2.line(canvas,(10, y_size-30),(10, y_size-30-6), (0,0,0), thickness) 
        cv2.line(canvas,(10+int(legend_length*self.resolution), y_size-30-6),(int(10+legend_length*self.resolution), y_size-30), (0,0,0), thickness) 
        text = "1m"
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(canvas, text, (10, y_size-14), font, 0.4, (0,0,0), thickness,cv2.LINE_AA)

        if static:
            plt.imshow(canvas)
            plt.show()
        else:
            canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
            cv2.imshow('Simulation',canvas)
            cv2.waitKey(int(self.time_interval*1000))

        return

if __name__ == "__main__":
# DEMO for testing SLAM
    sim = Simulator()
    slam = TestSlam(simulation=True,simulator=sim)
    success_count = 0
    n_iteration = 2
    for i in range(n_iteration):
        sim.slam_test_prep(slam)
        sim.test_slam()
        if (slam.real_err[-1]<1):
            success_count += 1
        print("Sim "+str(i)+", "+str(slam.real_err[-1]<1))
        #slam.debug_trend()
    print("sim complete")
    print("success rate = "+str(success_count/float(n_iteration)))

    '''
# DEMO for testing a Data associator
    sim = Simulator()
    da = DataAssociator(simulation=True);

    n_attempts = 1
    t = time()
    sim.init_da_statistics()
    for i in range(n_attempts):
#prepare landmarks, observations, etc
        sim.data_associator_test_prep()
# retrieve sim.Z, sim.X, calculate the prediction list 
# (in current implementation it's in format of self.association(see format near definition)), 
# ideally, the prediction calculated should be identical to sim.true_association
        association = da.associate(sim.X,sim.Z)
        if (i % 100 == 0):
            # show progress
            print(float(i)/n_attempts)
        # upon setting evaluate_da() will trigger visualization when sth goes wrong
        sim.evaluate_da(association, visualize_at_err = False)
        #sim.visualize()

    t = time()-t
    t /= n_attempts
    sim.show_da_statistics(da)

    # not really accurate... just for ref
    print("Freq = " + str(1/t))
    '''
