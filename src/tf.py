import numpy as np
from math import atan2,radians,degrees,sin,cos,pi,tan,copysign,asin,acos,isnan

# for coordinate transformation
# 3 coord frame:
# 1. Inertial, world frame, as in vicon
# 2. Track frame, local world frame, used in RCPtrack as world frame
# 3. Car frame, origin at rear axle center, y pointing forward, x to right
class TF:
    def __init__(self):
        pass

    def euler2q(self,roll,pitch,yaw):
        q = np.array([ cos(roll/2)*cos(pitch/2)*cos(yaw/2)+sin(roll/2)*sin(pitch/2)*sin(yaw/2), -cos(roll/2)*sin(pitch/2)*sin(yaw/2)+cos(pitch/2)*cos(yaw/2)*sin(roll/2), cos(roll/2)*cos(yaw/2)*sin(pitch/2) + sin(roll/2)*cos(pitch/2)*sin(yaw/2), cos(roll/2)*cos(pitch/2)*sin(yaw/2) - sin(roll/2)*cos(yaw/2)*sin(pitch/2)])
        return q

    def q2euler(self,q):
        R = self.q2R(q)
        roll,pitch,yaw = self.R2euler(R)
        return (roll,pitch,yaw)

    # given unit quaternion, find corresponding rotation matrix (passive)
    def q2R(self,q):
        #assert(isclose(np.linalg.norm(q),1,atol=0.001))
        Rq = [[q[0]**2+q[1]**2-q[2]**2-q[3]**2, 2*q[1]*q[2]+2*q[0]*q[3], 2*q[1]*q[3]-2*q[0]*q[2]],\
           [2*q[1]*q[2]-2*q[0]*q[3],  q[0]**2-q[1]**2+q[2]**2-q[3]**2,    2*q[2]*q[3]+2*q[0]*q[1]],\
           [2*q[1]*q[3]+2*q[0]*q[2],  2*q[2]*q[3]-2*q[0]*q[1], q[0]**2-q[1]**2-q[2]**2+q[3]**2]]
        Rq = np.matrix(Rq)
        return Rq

    # given euler angles, find corresponding rotation matrix (passive)
    # roll, pitch, yaw, (in reverse sequence, yaw is applied first, then pitch applied to intermediate frame)
    # all in radians
    def euler2R(self, roll,pitch,yaw):
        '''
        R = [[ c2*c3, c2*s3, -s2],
        ...  [s1*s2*s3-c1*s3, s1*s2*s3+c1*c3, c2*s1],
        ...  [c1*s2*c3+s1*s3, c1*s2*s3-s1*c3, c2*c1]]
        '''

        R = [[ cos(pitch)*cos(yaw), cos(pitch)*sin(yaw), -sin(pitch)],\
          [sin(roll)*sin(pitch)*sin(yaw)-cos(roll)*sin(yaw), sin(roll)*sin(pitch)*sin(yaw)+cos(roll)*cos(yaw), cos(pitch)*sin(roll)],\
          [cos(roll)*sin(pitch)*cos(yaw)+sin(roll)*sin(yaw), cos(roll)*sin(pitch)*sin(yaw)-sin(roll)*cos(yaw), cos(pitch)*cos(roll)]]
        R = np.matrix(R)
        return R
        
    # same as euler2R, rotation order is different, roll, pitch, yaw, in that order
    # degree in radians
    def euler2Rxyz(self, roll,pitch,yaw):
        '''
        Rx = [[1,0,0],[0,c1,s1],[0,-s1,c1]]
        Ry = [[c1,0,-s1],[0,1,0],[s1,0,c1]]
        Rz = [[c1,s1,0],[-s1,c1,0],[0,0,1]]
        '''
        Rx = np.matrix([[1,0,0],[0,cos(roll),sin(roll)],[0,-sin(roll),cos(roll)]])
        Ry = np.matrix([[cos(pitch),0,-sin(pitch)],[0,1,0],[sin(pitch),0,cos(pitch)]])
        Rz = np.matrix([[cos(yaw),sin(yaw),0],[-sin(yaw),cos(yaw),0],[0,0,1]])
        R = Rz*Ry*Rx
        return R

    # euler angle from R, in rad, roll,pitch,yaw
    def R2euler(self,R):
        roll = atan2(R[1,2],R[2,2])
        pitch = -asin(R[0,2])
        yaw = atan2(R[0,1],R[0,0])
        return (roll,pitch,yaw)
    # given pose of T(track frame) in W(vicon world frame), and pose of B(car body frame) in W,
    # find pose of B in T
    # T = [q,x,y,z], (7,) np.array
    # everything in W frame unless noted, vec_B means in B basis, e.g.
    # a_R_b denotes a passive rotation matrix that transforms from b to a
    # vec_a = a_R_b * vec_b
    def reframe(self,T, B):
        # TB = OB - OT
        OB = np.matrix(B[-3:]).T
        OT = np.matrix(T[-3:]).T
        TB = OB - OT
        T_R_W = self.q2R(T[:4])
        B_R_W = self.q2R(B[:4])

        # coord of B origin in T, in T basis
        TB_T = T_R_W * TB
        # in case we want full pose, just get quaternion from the rotation matrix below
        B_R_T = B_R_W * np.linalg.inv(T_R_W)
        (roll,pitch,yaw) = self.R2euler(B_R_T)

        # x,y, heading
        return (TB_T[0,0],TB_T[1,0],yaw+pi/2)
# reframe, using translation and R(passive)
    def reframeR(self,T, x,y,z,R):
        # TB = OB - OT
        OB = np.matrix([x,y,z]).T
        OT = np.matrix(T[-3:]).T
        TB = OB - OT
        T_R_W = self.q2R(T[:4])
        B_R_W = R

        # coord of B origin in T, in T basis
        TB_T = T_R_W * TB
        # in case we want full pose, just get quaternion from the rotation matrix below
        B_R_T = B_R_W * np.linalg.inv(T_R_W)
        (roll,pitch,yaw) = self.R2euler(B_R_T)

        # x,y, heading
        return (TB_T[0,0],TB_T[1,0],yaw+pi/2)

