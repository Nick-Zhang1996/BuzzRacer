import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.stats import norm

def column_wise_norm(x):
    return np.sum(x**2, axis=0) ** 0.5

def dist(x1, y1, x2, y2):
    return ((x1-x2)**2 + (y1-y2)**2) ** 0.5

def dists(x, y, xys):
    diff = xys - np.array([[x, y]]).T
    return column_wise_norm(diff)

def path_length(xys):
    dxy = xys[:, 1:] - xys[:, :-1]
    return np.sum(column_wise_norm(dxy))

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

        self.spline_x = None
        self.spline_y = None
        self.racing_line_timestep = 1.0

        self.x_min = np.min(self.features[0])
        self.x_max = np.max(self.features[0])
        self.y_min = np.min(self.features[1])
        self.y_max = np.max(self.features[1])

    def save(self, path):
        with open(path, 'w') as f:
            pickle.dump(self, f, protocol=2)

    @classmethod
    def load_file(cls, path):
        with open(path, 'r') as f:
            return pickle.load(f)

    def draw(self, show=True):
        plt.plot(self.features[0], self.features[1], 'rx')
        plt.plot([x for x,y in self.border1], [y for x,y in self.border1], 'k-')
        plt.plot([x for x,y in self.border2], [y for x,y in self.border2], 'k-')
        cmap = plt.cm.get_cmap('plasma')
        if self.spline_x is not None:
            lap_time = self.racing_line.shape[1] * self.racing_line_timestep
            ts = np.linspace(0, lap_time, 200)
            x = self.spline_x(ts)
            y = self.spline_y(ts)
            v = column_wise_norm(np.stack([self.spline_x(ts,1), self.spline_y(ts,1)]))
            c = cmap(v / 2)
            for i in xrange(len(x)):
                plt.plot(x[i:i+2], y[i:i+2], color=c[i], lw=5)

        plt.axis('equal')
        if show:
            plt.show()

    def wall_dists(self, poses):
        assert poses.shape[0] == 2 and poses.ndim == 2

        distances = np.ones(poses.shape[1]) * (self.x_max + self.y_max - self.x_min - self.y_min)

        for border in [self.border1, self.border2]:
            for (x1, y1), (x2, y2) in zip(border[:-1], border[1:]):
                border_dist = dist(x1, y1, x2, y2)
                if border_dist < 1e-5:
                    new_dists = dists(x1, y1, poses)
                    distances = np.minimum(distances, new_dists)
                else:
                    v = np.array([x2-x1, y2-y1]) / border_dist
                    normal = np.array([-v[1], v[0]])
                    rel_poses = poses - np.array([[x1, y1]]).T
                    line_proj = np.dot(v, rel_poses)
                    eps = 1e-5
                    valid_mask = np.logical_and(line_proj>-eps, line_proj<border_dist+eps)
                    new_dists = np.abs(np.dot(normal, rel_poses[:, valid_mask]))
                    distances[valid_mask] = np.minimum(distances[valid_mask], new_dists)

        return distances

    def collision_check(self, poses, radius):
        return self.wall_dists(poses) < radius

    def opt_racing_line(self, iterations, accel_limit):
        def cost(line, t_step):
            x, y = line
            cost = 0

            t = np.arange(len(x)) * float(t_step)
            self.spline_x = CubicSpline(t, x, bc_type='periodic')
            self.spline_y = CubicSpline(t, y, bc_type='periodic')

            t = np.linspace(0, t[-1], 1000)
            x = self.spline_x(t, 0)
            y = self.spline_y(t, 0)
            vx = self.spline_x(t, 1)  # velocity over time
            vy = self.spline_y(t, 1)
            ax = self.spline_x(t, 2)  # acceleration
            ay = self.spline_y(t, 2)
            jx = self.spline_x(t, 3)  # jerk
            jy = self.spline_y(t, 3)

            cost += t[-1]

            accel_sqr = ax*ax + ay*ay
            max_accel_sqr = np.max(accel_sqr)
            if max_accel_sqr > accel_limit**2:
                cost += 1000 * (max_accel_sqr - accel_limit**2)

            # apply a small bias towards smooth paths
            cost += 1e-5 * np.sum(accel_sqr)
            cost += 1e-5 * np.sum(jx*jx + jy*jy)

            # avoid collisions
            wall_dists = self.wall_dists(np.stack([x,y])).astype(np.float)
            min_dist = np.min(wall_dists)
            if min_dist < 0.1:
                cost += (0.1 - min_dist) * 1000

            # curvature limits
            allowed_curvature = 1.0 / 0.1  # 1 / min turn radius
            v = np.stack([vx, vy])
            unit_tangents = v / column_wise_norm(v)
            ds = v * self.racing_line_timestep
            curvature = column_wise_norm(unit_tangents / ds)
            max_curvature = np.max(curvature)
            if max_curvature > allowed_curvature:
                cost += 1000 * (max_curvature - allowed_curvature)

            return cost


        noise_scale = np.array([[1e-4]*2 + [1e-5]]).T
        batch_size = 50
        for iteration in xrange(iterations):
            noises = np.zeros((3, batch_size, self.racing_line.shape[1]))
            costs = np.zeros(batch_size)
            for i in xrange(batch_size):
                noise = np.random.normal(0, noise_scale, (3, noises.shape[2]))
                noise[2, :] = noise[2, 0]
                noise[0, (0,-1)] = 0
                noise[1, -1] = noise[1, 0]
                noises[:, i, :] = noise
                costs[i] = cost(self.racing_line + noise[:2],
                                self.racing_line_timestep + noise[2, 0])

            costs = (costs - np.mean(costs)) / max(np.std(costs), 1e-5)
            weights = np.exp(-costs)
            grad = np.dot(weights, noises) / np.sum(weights)
            time_grad = grad[2, 0]
            grad = grad[:2]

            last_cost = cost(self.racing_line, self.racing_line_timestep)
            init_t = t = 1.0
            new_cost = cost(self.racing_line + t * grad,
                            self.racing_line_timestep + t * time_grad)
            while t == init_t or new_cost <= last_cost:
                self.racing_line += t * grad
                self.racing_line_timestep += t * time_grad
                t += init_t
                last_cost = new_cost
                new_cost = cost(self.racing_line + t*grad,
                                self.racing_line_timestep + t*time_grad)

            total_time = self.racing_line_timestep * (self.racing_line.shape[1] - 1)
            print "cost", np.round(last_cost, 6), "lap time", np.round(total_time, 3)

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

            ctrl_res = 1.0
            new_controls = np.stack([np.arange(ctrl_res)/ctrl_res]*2, axis=1) * dpos + pos
            control_points += [(x,y) for x,y in new_controls]

            pos += dpos
            theta += dtheta_map[c]

        return cls(features_list, left_border, right_border, control_points[1:])


if __name__ == '__main__':
    DIR = os.path.dirname(__file__)
    save_path = os.path.join(DIR, 'track_mk111.pkl')

    track = Track.load('ssrrsllsrrssrsllsrrssrss', 0.565)
    if os.path.exists(save_path):
        preopt = Track.load_file(save_path)
        track.racing_line = preopt.racing_line
        track.racing_line_timestep = preopt.racing_line_timestep
        track.spline_x = preopt.spline_x
        track.spline_y = preopt.spline_y

    # print track.wall_dists(np.array([[0., 0.]]).T)
    # print track.wall_dists(np.array([[-1, -1.4]]).T)

    # track.draw()

    for _ in xrange(5000):
        track.opt_racing_line(10, 3.0)

        plt.clf()
        track.draw(False)
        plt.pause(0.01)

        track.save(save_path)

    print "done"
