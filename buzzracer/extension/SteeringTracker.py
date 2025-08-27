''' Measure steering angle using optitrack '''

import numpy as np

from buzzracer.extension.Extension import Extension


class SteeringTracker(Extension):
    def __init__(self):
        Extension.__init__(self,'steering_tracker')
        self.pos_vec = None

        self.vi = None
        ''' Visual tracking system instance'''

        self.measured_steering = 0.0
        self.steering_history = []
        self.main.cars[0].debug_dict['measured_steering'] = self.steering_history

    def init(self):
        self.vi = self.main.vi
        self.vi.streamingClient.labeled_marker_listener = self.labeled_marker_listener

    def labeled_marker_listener(self, pos_vec):
        ''' Track labeled marker placed on steering rack'''
        # dim: -x,z,y
        self.pos_vec = np.array(pos_vec)

    def update(self):
        if self.pos_vec is None:
            return
        # filter down to markers close to car
        car = self.main.cars[0]
        states = car.states
        x, y, heading, _, _, _ = states
        dx = -self.pos_vec[:, 0] - x
        dy = self.pos_vec[:, 2] - y
        dists = (dx**2+dy**2)**0.5
        mask = dists < 0.2
        pos_vec = self.pos_vec[mask, :]
        pos_vec = np.vstack([-pos_vec[:, 0], pos_vec[:, 2]]).T
        if pos_vec.shape[0] != 2:
            self.print_warning("can't determine steering, too many markers")
            self.measured_steering = 0.0
            self.steering_history.append(self.measured_steering)
            return
        # pos_vec now contains two balls that indicate right wheel steering angle
        dx = pos_vec[1, 0] - pos_vec[0, 0]
        dy = pos_vec[1, 1] - pos_vec[0, 1]
        angle = np.arctan2(dy, dx)
        self.measured_steering = (angle-heading+3*np.pi) % (2*np.pi)-np.pi
        self.steering_history.append(self.measured_steering)
