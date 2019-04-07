import numpy as np
import tf

class CameraModel(object):
    def __init__(self, angle_down, height, focal_length, img_width, img_height):
        self.angle_down = angle_down
        self.height = height
        self.focal_length = focal_length
        self.img_width = img_width
        self.img_height = img_height

        self.tf_base_cam = np.dot(
            tf.transformations.rotation_matrix(-self.angle_down, (0,1,0)),
            tf.transformations.translation_matrix((0, 0, -self.height))
        )

        print "Horizontal FOV is", 2 * np.math.atan2(self.img_width, 2*self.focal_length)

    def project_closest_onto_image_multiple(self, robot_poses, world_features):
        # assert world_features.shape[0] == 4 and all(world_features[3] == 1.0)

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

        # print tf_world_base.shape

        full_tf = np.dot(self.tf_base_cam, tf_world_base)

        # print full_tf.shape
        # print full_tf[:,0].round(2)

        cam_feats = np.dot(full_tf, world_features)

        # print cam_feats.shape
        # print cam_feats[:, 0]

        img_feats = np.stack([
            cam_feats[1] / cam_feats[0] * -self.focal_length + self.img_width / 2,
            cam_feats[2] / cam_feats[0] * -self.focal_length + self.img_height / 2
        ], axis=1)

        # print img_feats.shape

        out = []
        for i in xrange(img_feats.shape[0]):
            mask_in_img = cam_feats[0, i] > 0
            mask_in_img = np.logical_and(mask_in_img, img_feats[i, 0] > 0)
            mask_in_img = np.logical_and(mask_in_img, img_feats[i, 0] < self.img_width)
            mask_in_img = np.logical_and(mask_in_img, img_feats[i, 1] > 0)
            mask_in_img = np.logical_and(mask_in_img, img_feats[i, 1] < self.img_height)

            n = np.count_nonzero(mask_in_img)
            if n > 0:
                j = np.argmin(cam_feats[0, i])
                out.append(img_feats[i, :, j])
            else:
                out.append(None)

        return out
