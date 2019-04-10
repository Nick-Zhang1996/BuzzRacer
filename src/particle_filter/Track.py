import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

def dist(x1, y1, x2, y2):
    return ((x1-x2)**2 + (y1-y2)**2) ** 0.5

def dists(x, y, xys):
    diff = xys - np.array([[x, y]]).T
    return np.sum(diff*diff, axis=0) ** 0.5

def path_length(xys):
    dx = xys[0, 1:] - xys[0, :-1]
    dy = xys[1, 1:] - xys[1, :-1]
    return sum((dx*dx + dy*dy) ** 0.5)

"""
Track contains a list of features. Each feature is a point in
homogeneous 3D coordinates (homongenous for tf compatibility)
"""
class Track(object):
    def __init__(self, features_list, border1, border2, control_points):
        # features list is x, y pairs in track space
        self.features = np.zeros((4, len(features_list)), dtype=np.float)
        for i, (x, y) in enumerate(features_list):
            self.features[:, i] = [x, y, 0, 1]

        self.border1 = list(border1) + [border1[0]]
        self.border2 = list(border2) + [border2[0]]

        control_points = [(0,0)] + list(control_points) + [(0,0)]
        self.racing_line = np.zeros((2, len(control_points)), dtype=np.float)
        for i, (x, y) in enumerate(control_points):
            self.racing_line[:, i] = [x, y]

        default_speed = 0.5
        lap_time_guess = float(path_length(self.racing_line)) / default_speed
        self.racing_line_timestep = lap_time_guess / self.racing_line.shape[1]

        self.x_min = np.min(self.features[0])
        self.x_max = np.max(self.features[0])
        self.y_min = np.min(self.features[1])
        self.y_max = np.max(self.features[1])

    def draw(self, show=True):
        plt.plot(self.features[0], self.features[1], 'rx')
        plt.plot([x for x,y in self.border1], [y for x,y in self.border1], 'k-')
        plt.plot([x for x,y in self.border2], [y for x,y in self.border2], 'k-')
        plt.plot(self.racing_line[0], self.racing_line[1], 'bo-')
        plt.axis('equal')
        if show:
            plt.show()

    def collision_check(self, poses, radius):
        assert poses.shape[0] >= 2 and poses.ndim == 2

        collisions = np.zeros(poses.shape[1], dtype=np.bool)

        for border in [self.border1, self.border2]:
            for (x1, y1), (x2, y2) in zip(border[:-1], border[1:]):
                border_dist = dist(x1, y1, x2, y2)
                if border_dist < 1e-5:
                    # border points are the same, treat as one point
                    new_collisions = (dists(x1, y1, poses[:2]) < radius)
                    collisions = np.logical_or(collisions, new_collisions)
                else:
                    v = np.array([x2-x1, y2-y1]) / border_dist
                    normal = np.array([-v[1], v[0]])
                    rel_poses = poses[:2] - np.array([[x1, y1]]).T
                    line_proj = np.dot(v, rel_poses)
                    normal_proj = np.dot(normal, rel_poses)
                    closest_here = np.logical_and(line_proj>0, line_proj<border_dist)
                    close_enough = (np.abs(normal_proj) < radius)
                    new_collisions = np.logical_and(closest_here, close_enough)
                    collisions = np.logical_or(collisions, new_collisions)

        return collisions

    def opt_racing_line(self, iterations):
        def cost(line, t_step):
            x, y = line

            t = np.arange(len(x)) * float(t_step)
            spline_x = CubicSpline(t, x, bc_type='periodic')
            spline_y = CubicSpline(t, y, bc_type='periodic')
            vx = spline_x(t, 1)  # velocity over time
            vy = spline_y(t, 1)
            jx = spline_x(t, 3)  # jerk over time
            jy = spline_y(t, 3)

            speeds = (vx*vx + vy*vy) ** 0.5
            speed_error = (2.0 - speeds) ** 2

            cost = np.sum(speed_error) + 0.001 * np.sum(jx*jx + jy*jy)

            if np.any(self.collision_check(line, 0.1)):
                cost += 100

            return cost

        last_cost = cost(self.racing_line, self.racing_line_timestep)
        # noise_scale = np.array([[0.01, 0.01]]).T
        noise_scale = [0.01, 0.01]
        for iteration in xrange(iterations):
            # noise = np.random.normal(0, noise_scale, self.racing_line.shape)
            # new_racing_line = self.racing_line + noise
            new_timestep = self.racing_line_timestep + np.random.normal(0, 0.002)
            noise = np.random.normal(0, noise_scale)
            new_racing_line = self.racing_line.copy()
            i = np.random.randint(self.racing_line.shape[1])
            new_racing_line[:, i] += noise
            new_racing_line[0, (0,-1)] = 0
            new_racing_line[1, (0,-1)] = np.mean(new_racing_line[1, (0,-1)])
            new_cost = cost(new_racing_line, new_timestep)
            if new_cost < last_cost:
                self.racing_line = new_racing_line
                self.racing_line_timestep = new_timestep
                last_cost = new_cost

                total_time = self.racing_line_timestep * (self.racing_line.shape[1] - 1)
                avg_speed = path_length(self.racing_line) / total_time
                print "cost", np.round(new_cost, 3), "avg speed", np.round(avg_speed, 3)

    """
    load
    Parse a description string. The string uses the character 's' for
    straight, 'r' for right turn, and 'l' for left turn.
    """
    @classmethod
    def load(cls, description_string, piece_size):
        assert all(c in 'srl' for c in description_string)

        dtheta_map = {'s': 0, 'r': -np.pi/2, 'l': np.pi/2}
        local_dpos_map = {'s': (piece_size, 0),
                          'r': (piece_size/2., -piece_size/2.),
                          'l': (piece_size/2., piece_size/2.)}

        # define location of features (centers of red areas) relative
        # to center of lane at entry
        turn_outer_feature_x = 0.95 * piece_size
        turn_outer_feature_y = 0.45 * piece_size
        turn_inner_feature_x = 0.05 * piece_size
        turn_inner_feature_y = 0.45 * piece_size
        feature_pos_from_piece_entry = {
            's': [],
            'r': [(turn_outer_feature_x, turn_outer_feature_y),
                  (turn_inner_feature_x, -turn_inner_feature_y)],
            'l': [(turn_outer_feature_x, -turn_outer_feature_y),
                  (turn_inner_feature_x, turn_inner_feature_y)]
        }

        features_list = []
        left_border = []
        right_border = []
        control_points = []
        pos = np.zeros(2, dtype=np.float)
        theta = 0
        for c in description_string:
            # features_list.append(tuple(pos))
            cos = np.cos(theta)
            sin = np.sin(theta)
            rot_mat = [[cos, -sin], [sin, cos]]

            for local_feature in feature_pos_from_piece_entry[c]:
                global_feature = pos + np.dot(rot_mat, local_feature)
                features_list.append(tuple(global_feature))

            for border, sgn, ch in [(left_border, 1, 'r'), (right_border, -1, 'l')]:
                local_borders = [(0, sgn*0.5*piece_size)]
                if c == ch:
                    local_borders.append((piece_size, sgn*0.5*piece_size))
                for b in local_borders:
                    border.append(tuple(pos + np.dot(rot_mat, b)))

            dpos = np.dot(rot_mat, local_dpos_map[c])

            new_controls = np.stack([np.arange(2)/2.]*2, axis=1) * dpos + pos
            control_points += [(x,y) for x,y in new_controls]

            pos += dpos
            theta += dtheta_map[c]

        return cls(features_list, left_border, right_border, control_points[1:])


if __name__ == '__main__':
    track = Track.load('ssrrsllsrrssrsllsrrssrss', 0.565)
    track.opt_racing_line(10000)
    track.draw()
