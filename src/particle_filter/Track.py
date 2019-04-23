import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline


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


class Track(object):
    """
    Track:
    - list of visual features in world space
    - collision checking
    - pre-computed racing line optimization
    """
    def __init__(self, features_list, border1, border2, control_points, desired_lap_time):
        # features list is x, y pairs in track space
        self.features = np.zeros((4, len(features_list)), dtype=np.float)
        for i, (x, y) in enumerate(features_list):
            self.features[:, i] = [x, y, 0, 1]

        self.border1 = list(border1) + [border1[0]]
        self.border2 = list(border2) + [border2[0]]

        control_points = [(0, 0)] + list(control_points) + [(0, 0)]
        self.racing_line = np.zeros((3, len(control_points)), dtype=np.float)
        time_per_point = float(desired_lap_time) / (len(control_points) - 1)
        for i, (x, y) in enumerate(control_points):
            self.racing_line[:, i] = [x, y, time_per_point]

        self.spline_x = None
        self.spline_y = None

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
        plt.plot([x for x, y in self.border1], [y for x, y in self.border1], 'k-')
        plt.plot([x for x, y in self.border2], [y for x, y in self.border2], 'k-')
        cmap = plt.cm.get_cmap('plasma')
        if self.spline_x is not None:
            lap_time = sum(self.racing_line[2, :])
            ts = np.linspace(0, lap_time, 200)
            x = self.spline_x(ts)
            y = self.spline_y(ts)
            v = column_wise_norm(np.stack([self.spline_x(ts, 1), self.spline_y(ts, 1)]))
            c = cmap(v / 2)
            for i in xrange(len(x)):
                plt.plot(x[i:i+2], y[i:i+2], color=c[i], lw=5)
            plt.plot(self.racing_line[0], self.racing_line[1], 'ko')

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
                    valid_mask = np.logical_and(line_proj > -eps, line_proj < border_dist+eps)
                    new_dists = np.abs(np.dot(normal, rel_poses[:, valid_mask]))
                    distances[valid_mask] = np.minimum(distances[valid_mask], new_dists)

        return distances

    def collision_check(self, poses, radius):
        return self.wall_dists(poses) < radius

    def cost_fn(self, line, accel_limit):
        x, y, dt = line
        cost = 0

        t = np.cumsum([0] + dt.tolist()[:-1])
        self.spline_x = CubicSpline(t, x, bc_type='periodic')
        self.spline_y = CubicSpline(t, y, bc_type='periodic')

        t = np.linspace(0, t[-1], 500)
        x = self.spline_x(t, 0)
        y = self.spline_y(t, 0)
        vx = self.spline_x(t, 1)  # velocity over time
        vy = self.spline_y(t, 1)
        ax = self.spline_x(t, 2)  # acceleration
        ay = self.spline_y(t, 2)
        # jx = self.spline_x(t, 3)  # jerk
        # jy = self.spline_y(t, 3)

        cost += t[-1]
        # print "time cost", cost

        accel_sqr = ax*ax + ay*ay
        max_accel = np.max(accel_sqr) ** 0.5
        if max_accel > accel_limit:
            cost += 100 * (max_accel - accel_limit)
            # print "accel cost", 1000 * (max_accel - accel_limit)

        # apply a small bias towards smooth paths
        # cost += 1e-5 * np.sum(accel_sqr)
        # cost += 1e-5 * np.sum(jx*jx + jy*jy)
        # print "smoothness cost", 1e-5 * (np.sum(accel_sqr) + np.sum(jx*jx + jy*jy))

        # apply a bias toward good spacing
        cost += 1 * np.sum(column_wise_norm(line[:2, 1:] - line[:2, :-1]) ** 2)

        # avoid collisions
        wall_dists = self.wall_dists(np.stack([x,y])).astype(np.float)
        min_dist = np.min(wall_dists)
        if min_dist < 0.1:
            cost += (0.1 - min_dist) * 100
            # print "collision cost", (0.1 - min_dist) * 1000

        # curvature limits
        allowed_curvature = 1.0 / 0.1  # 1 / min turn radius
        v = np.stack([vx, vy])
        speed = column_wise_norm(v)
        unit_tangents = v / speed
        total_time = np.sum(self.racing_line[2, :-1])
        ds = speed[:-1] * (total_time / v.shape[1])
        dT = unit_tangents[:, 1:] - unit_tangents[:, :-1]
        # print "dT/ds", dT[:, 100] / ds[100]
        curvature = column_wise_norm(dT / ds)
        # print curvature.round()
        max_curvature = np.max(curvature)
        if max_curvature > allowed_curvature:
            cost += 100 * (max_curvature - allowed_curvature)
            # print "curvature cost", 1000 * (max_curvature - allowed_curvature)

        return cost

    def opt_racing_line(self, iterations, accel_limit, grad_update_norm):
        def cost(line):
            return self.cost_fn(line, accel_limit)

        perturbance = 0.0001
        for iteration in xrange(iterations):
            orig_cost = cost(self.racing_line)
            grad = np.zeros_like(self.racing_line[:, :-1])
            for j in xrange(grad.shape[1]):
                for axis in range(3):
                    new_line = self.racing_line.copy()
                    sgn = 1 if np.random.random() < 0.5 else -1
                    new_line[axis, j] += sgn * perturbance
                    new_line[:, -1] = new_line[:, 0]
                    grad[axis, j] = sgn * (cost(new_line) - orig_cost) / perturbance

            grad = np.concatenate([grad, grad[:, 0].reshape(3, 1)], axis=1)

            norm = np.sum(grad ** 2) ** 0.5
            if norm > grad_update_norm:
                grad *= grad_update_norm / norm
            # print np.sum(grad ** 2) ** 0.5

            self.racing_line -= grad
            self.racing_line[2, self.racing_line[2] <= 0] = 0.01

            total_time = np.sum(self.racing_line[2, :-1])
            print "cost", np.round(cost(self.racing_line), 6), "lap time", np.round(total_time, 3)

    """
    load
    Parse a description string. The string uses the character 's' for
    straight, 'r' for right turn, and 'l' for left turn.
    """
    @classmethod
    def load(cls, description_string, piece_size, control_resolution):
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

            spacing = np.arange(control_resolution) / float(control_resolution)
            new_controls = np.stack([spacing]*2, axis=1) * dpos + pos
            control_points += [(x, y) for x, y in new_controls]

            pos += dpos
            theta += dtheta_map[c]

        return cls(features_list, left_border, right_border, control_points[1:],
                   0.4 * len(description_string))


if __name__ == '__main__':
    DIR = os.path.dirname(__file__)
    save_path = os.path.join(DIR, 'track_mk111.pkl')

    if os.path.exists(save_path):
        track = Track.load_file(save_path)
    else:
        track = Track.load('ssrrsllsrrssrsllsrrssrss', 0.565, 1)

    max_accel = 3.0
    best_cost = track.cost_fn(track.racing_line, max_accel)
    costs = []
    for i in xrange(5000):
        track.opt_racing_line(1, max_accel, 0.0005)

        cost = track.cost_fn(track.racing_line, max_accel)
        costs.append(cost)
        if cost < best_cost:
            track.save(save_path)
            best_cost = cost

        if i % 10 == 0:
            plt.clf()
            track.draw(False)
            # plt.plot(costs)
            # plt.plot([best_cost]*len(costs))
            plt.pause(0.01)

    print "done"
