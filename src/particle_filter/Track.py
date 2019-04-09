import numpy as np
import matplotlib.pyplot as plt

"""
Track contains a list of features. Each feature is a point in
homogeneous 3D coordinates (homongenous for tf compatibility)
"""
class Track(object):
    def __init__(self, features_list):
        # features list is x, y pairs in track space
        self.features = np.zeros((4, len(features_list)), dtype=np.float)
        for i, (x, y) in enumerate(features_list):
            self.features[:, i] = [x, y, 0, 1]

        self.x_min = np.min(self.features[0])
        self.x_max = np.max(self.features[0])
        self.y_min = np.min(self.features[1])
        self.y_max = np.max(self.features[1])

    def draw(self, show=True):
        plt.plot(self.features[0], self.features[1], 'rx')
        plt.axis('equal')
        if show:
            plt.show()

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

        theta = 0
        features_list = []
        bounds = []
        pos = np.zeros(2, dtype=np.float)
        for c in description_string:
            # features_list.append(tuple(pos))
            cos = np.cos(theta)
            sin = np.sin(theta)
            rot_mat = [[cos, -sin], [sin, cos]]

            for local_feature in feature_pos_from_piece_entry[c]:
                global_feature = pos + np.dot(rot_mat, local_feature)
                features_list.append(tuple(global_feature))

            dpos = np.dot(rot_mat, local_dpos_map[c])
            pos += dpos
            theta += dtheta_map[c]

        return cls(features_list)


if __name__ == '__main__':
    track = Track.load('ssrrsllsrrssrsllsrrssrss', 0.565)
    track.draw()
