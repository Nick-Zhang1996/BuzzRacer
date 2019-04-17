#!/usr/bin/python

# Author: Nick Zhang, Fall 2018, nickzhang@gatech.edu
# This is a dataAssociator that maps feature to landmarks
# more into to be added on source paper, etc
# XXX current bug: not checking complete graph , multiple mapping for one feature
# rotation is not from center

import numpy as np
import rospy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from collections import namedtuple
from math import sqrt, radians
from time import time

Association = namedtuple("Association", "feature landmark edges")

class DataAssociator:

    def __init__(self, simulation = False):
        self.simulation = simulation

        if not simulation:
            self.marker_asso_pub = rospy.Publisher('~landmark_asso', Marker, queue_size=1)
        # screening distance for nearest neighbour
# for ~96% success, set to 4*stddev(angleErr*maxRange+robotErr+featureErr+landmarkErr)
        #self.nn_distance = 4*(radians(7.0)*20.0+7+1+0.0)
        self.nn_distance = 5.0
# max error in distance for two associations to be compatible
# for ~96% success, set to 4*stddev(featureErr), relative to vehicle
        self.joint_distance = 2.0
        self.clear_distance = 2.0

    """
    Given X(states) and Z(observation), find the best data association H
    X: [x, y, theta, x1, y1, x2, y2, ...]^T shape = (2*n_landmark+3, 1)
    Z: [zx1, zy1, zx2, zy2, ...]  shape = (2*n_feature)
    ideally, X = HZ 

    In current implementation, the function does not return the true sparse matrix H
    return: association, a two column matrix, the 0th coloum is index of feature in Z
    the 1st coloum of landmark in X. 

    This is in accordence with laserSLAM.py 
    Note, unlike in laserSLAM.py, these index I use are not index of specific matrix entries, the entries are:
    X[3+id+x*2:+2,0], and Z[id+x*2:+2], respectively. 

    If a feature is believed to be a new landmark, it will be assigned -1 for landmark id
    If a feature does not show up in association, it's not fit to be used for any purpose
    """
    def associate(self, X, Z ):

# Create a graph with each association pair (one feature, one landmark) as a node, then edges are drawn between compatible nodes

# For each feature, find potential landmark association with nearest neighbour
# and create a node for each association pair
        self.X = X
        self.Z = Z
        id_z = list(range(0, len(Z)/2))
        self.nodes = []
        for i in id_z:
            Z_x = Z[i*2]
            Z_y = Z[i*2+1]
            id_x_list = list(range(0, (len(X)-3)/2))
            # for NN, we check dx and dy separately, because this is a rough screening, and we want to be efficient
            id_x_list = filter(
                    lambda index: abs(Z_x - X[3+2*index]) < self.nn_distance and abs(Z_y - X[4+2*index]) < self.nn_distance, \
                    id_x_list )
            self.add_node(i,id_x_list)

# if dist(feature1,feature2) and dist(landmark1, landmark2) are nearly equal, 1 and 2 are compatible
# no edge will be drawn for landmark candidates for same feature 
        n_node = len(self.nodes)
        for i in range(n_node):
            self.nodes[i].edges.append(i)
            for j in range(i+1,n_node):
                if self.is_compatible(i,j):
                    self.nodes[i].edges.append(j)
                    self.nodes[j].edges.append(i)
                 
# find a complete subgraph with most nodes
# start with the node with most edges
        keys = [len(node.edges) for node in self.nodes]
        order = np.argsort(keys)
        order = list(order[::-1])
        
# current number of nodes in max clique
        max_clique = []
        for start_node in order:
            count = len(self.nodes[start_node].edges)
            if count < len(max_clique):
                #print(count,len(max_clique))
# the remaining nodes can't form a larger clique
                break
# greedy, start with current node(max edges, likely to be in max clique)
# NOTE this does not guarantee the best solution
            clique = [start_node]

            remain = list(order)
            remain.remove(start_node)
            depleted = False
            while not depleted:
                depleted = True
# start with nodes with most edges
                for node in remain:
                    if (all([i in self.nodes[node].edges for i in clique])):
                        clique.append(node)
                        remain.remove(node)
                        depleted = False

# screening, if two features are too close to each other, ignore them
            
            if len(clique) > len(max_clique):
                max_clique = clique
             
        # Identify potential new landmarks, this will be futher filtered in LaserSLAM.py
# We do this before removing ambiguous features so we wouldn't identify them as new landmarks
        matched_features = [self.nodes[i].feature for i in max_clique]
        unmatched_features = filter(
                            lambda i : not i in matched_features,
                            id_z)
                                    
# remove features too close to each other, the feature detector shouldn't return close features, just to ensure
        i = 0
        while i<len(max_clique)-1:
            j = i+1
            while j<len(max_clique):
                if self.feature_distance(max_clique[i],max_clique[j]) < self.clear_distance:
                    node1 = max_clique[i]
                    node2 = max_clique[j]
                    #print(self.nodes[node1],self.nodes[node2])
                    #print(self.feature_distance(i,j))
                    max_clique.remove(node1)
                    max_clique.remove(node2)
                j += 1
            i += 1
# construct H/association from max clique
        self.association = [[self.nodes[node].feature, self.nodes[node].landmark] for node in max_clique]
        new_landmark = [[i, -1] for i in unmatched_features]
        for pair in new_landmark:
            self.association.append(pair)

        return None if self.association is None else np.array(self.association)

# add a node to graph (self.nodes)
# i: index for self.Z
# id_x_list: list of index for self.X
# note: not direct index
    def add_node(self,i,id_x_list):
        for id_x in id_x_list:
            node = Association(landmark = id_x, feature = i, edges = [])
            self.nodes.append(node)
            
# determine if two nodes are compatible
    def is_compatible(self, node1, node2):
        if (self.nodes[node1].landmark == self.nodes[node2].landmark or
            self.nodes[node1].feature == self.nodes[node2].feature):
            return False

        id1 = self.nodes[node1].landmark
        landmark1 = self.X[3+id1*2:3+id1*2+2,0]
        id2 = self.nodes[node2].landmark
        landmark2 = self.X[3+id2*2:3+id2*2+2,0]
        dist_landmark = DataAssociator.dist(landmark1, landmark2)

        id1 = self.nodes[node1].feature
        feature1 = self.Z[id1*2:id1*2+2]
        id2 = self.nodes[node2].feature
        feature2 = self.Z[id2*2:id2*2+2]
        dist_feature = DataAssociator.dist(feature1, feature2)

        return abs(dist_landmark - dist_feature) < self.joint_distance

    def feature_distance(self, node1, node2):
        id1 = self.nodes[node1].feature
        feature1 = self.Z[id1*2:id1*2+2]
        id2 = self.nodes[node2].feature
        feature2 = self.Z[id2*2:id2*2+2]
        return DataAssociator.dist(feature1, feature2)


# detail: use landmark index as index, c1 = [x1,y1] c2 = [x2,y2]
    @staticmethod
    def dist(c1,c2):
        return sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)



