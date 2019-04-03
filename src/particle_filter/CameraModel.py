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

    def project_onto_image(self, robot_pose, world_features, debug=False):
        # assert world_features.shape[0] == 4 and all(world_features[3] == 1.0)

        world_x, world_y, world_h = robot_pose[:3] * -1
        # tf_world_base = np.dot(
        #     tf.transformations.rotation_matrix(world_h, (0,0,1)), 
        #     tf.transformations.translation_matrix([world_x, world_y, 0, 1])
        # )

        c = np.cos(world_h)
        s = np.sin(world_h)
        tf_world_base = np.array([
            [c, -s, 0, world_x * c - world_y * s],
            [s, c, 0, world_x * s + world_y * c],
            [0, 0, 0, 0],
            [0, 0, 0, 1]
        ])

        # print "new\n", tf_world_base
        # print "old\n", tf_world_base_old

        full_tf = np.dot(self.tf_base_cam, tf_world_base)
        cam_feats = np.dot(full_tf, world_features)

        # get image point in x, y order
        img_feats = np.stack([
            cam_feats[1] / cam_feats[0] * -self.focal_length + self.img_width / 2,
            cam_feats[2] / cam_feats[0] * -self.focal_length + self.img_height / 2
        ])

        mask_in_img = cam_feats[0] > 0
        mask_in_img = np.logical_and(mask_in_img, img_feats[0] > 0)
        mask_in_img = np.logical_and(mask_in_img, img_feats[0] < self.img_width)
        mask_in_img = np.logical_and(mask_in_img, img_feats[1] > 0)
        mask_in_img = np.logical_and(mask_in_img, img_feats[1] < self.img_height)

        n = np.count_nonzero(mask_in_img)
        if n > 0:
            img_feats = img_feats[:, mask_in_img].round(0).astype(np.int)
            indices = np.arange(len(mask_in_img))[mask_in_img]
        else:
            img_feats = None
            indices = []

        if debug:
            print "car frame:\n", base_feats.round(2)[:2]
            print "camera frame:\n", cam_feats.round(2)[:3]
            print "img frame:\n", img_feats

        return img_feats, indices
