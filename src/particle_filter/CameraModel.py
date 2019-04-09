import numpy as np
import tf

class CameraModel(object):
    def __init__(self, angle_down, height, fov_horizontal, img_width, img_height):
        self.angle_down = angle_down
        self.height = height
        self.img_width = img_width
        self.img_height = img_height

        self.tf_base_cam = np.dot(
            tf.transformations.rotation_matrix(-self.angle_down, (0,1,0)),
            tf.transformations.translation_matrix((0, 0, -self.height))
        )

        self.focal_length = img_width / (2. * np.tan(fov_horizontal / 2.))

    """
    project_onto_image
    For each pose hypothesis in robot_poses, return a list of features that
    should be visible
    robot_poses: each column is a 2D pose (x, y, theta) shape=(3,n)
    world_features: track features as 3D homogenous coords shape=(4,m)
    return: List<Features> where Features is a shape (v,2) matrix of (x, y)
            visible features in image space, or Features is None if v=0
    """
    def project_onto_image(self, robot_poses, world_features):
        assert robot_poses.shape[0] >= 3 and np.all(world_features[3] == 1)

        world_x, world_y, world_h = robot_poses[:3, :] * -1

        c = np.cos(world_h)
        s = np.sin(world_h)
        zero = np.zeros_like(world_h)

        tf_world_base = np.stack([
            np.stack([c, -s, zero, world_x * c - world_y * s], axis=1),
            np.stack([s, c,  zero, world_x * s + world_y * c], axis=1),
            np.stack([zero, zero, zero, zero], axis=1),
            np.stack([zero, zero, zero, zero+1], axis=1)
        ], axis=1)

        full_tf = np.dot(self.tf_base_cam, tf_world_base)

        cam_feats = np.dot(full_tf, world_features)

        img_feats = np.stack([
            cam_feats[1] / cam_feats[0] * -self.focal_length + self.img_width / 2,
            cam_feats[2] / cam_feats[0] * -self.focal_length + self.img_height / 2
        ], axis=1)

        out = []
        for i in xrange(img_feats.shape[0]):
            mask_in_img = cam_feats[0, i] > 0
            mask_in_img = np.logical_and(mask_in_img, img_feats[i, 0] > 0)
            mask_in_img = np.logical_and(mask_in_img, img_feats[i, 0] < self.img_width)
            mask_in_img = np.logical_and(mask_in_img, img_feats[i, 1] > 0)
            mask_in_img = np.logical_and(mask_in_img, img_feats[i, 1] < self.img_height)

            n = np.count_nonzero(mask_in_img)
            if n > 0:
                out.append(img_feats[i, :, mask_in_img])
            else:
                out.append(None)

        return out
