import time

import matplotlib.pyplot as plt
import numpy as np

from particle_filter import CameraModel, Track, ParticleFilter


def main():
    features_list = [(-2, -2), (2, 2), (-1, 1), (0.5, -0.6), (0, 0)]
    track = Track(features_list)

    camera = CameraModel(angle_down=0.1, height=0.05, focal_length=200, 
                         img_width=640, img_height=480)

    pos_noise = 0.05
    yaw_noise = 0.1
    measurement_noise = 30.0
    false_positive_rate = 0.01
    false_negative_rate = 0.001
    noise_vec = [pos_noise, yaw_noise, measurement_noise, 
                 false_positive_rate, false_negative_rate]

    truth = ParticleFilter(1, [0, 0, 0, 0, 0], track, camera)
    filt = ParticleFilter(100, noise_vec, track, camera)

    truth.particles[:, 0] = [-0.2, -0.2, 0]
    for i in range(filt.particles.shape[1]):
        filt.particles[:, i] = truth.particles[:, 0] + np.random.normal(0, [0.1, 0.1, 0.2])
    # for i in range(filt.particles.shape[1]):
    #     filt.particles[2, i] = 0

    for t in range(1000):
        # print f.particles

        t0 = time.time()

        if t < 50:
            filt.predict(0, 0, 0.2)
            truth.predict(0, 0, 0.2)
        else:
            filt.predict(0.3, 0.3, 0.2)
            truth.predict(0.3, 0.3, 0.2)

        obs, _ = truth.camera.project_onto_image(truth.mean(1), track.features)
        if obs is None:
            obs = np.array([])
        obs += np.random.normal(0, 5, size=obs.shape).astype(np.int)
        obs = [tuple(f) for f in obs.T]
        # print(obs)
        filt.update_weights(obs)

        t1 = time.time()

        plt.clf()
        plt.scatter(filt.particles[0,:], filt.particles[1,:], color='b')

        x, y, h = truth.mean(1)
        plt.arrow(x, y, 0.2*np.cos(h), 0.2*np.sin(h), width=0.03, color="orange")

        x, y, h = filt.mean(0.5)
        std_x, std_y, _ = filt.stddev(0.8)
        is_converged = std_x < 0.05 and std_y < 0.05
        plt.arrow(x, y, 0.1*np.cos(h), 0.1*np.sin(h), width=0.03, 
                  color="g" if is_converged else "r")

        t2 = time.time()
        filt.resample()

        print time.time() - t2 + t1 - t0

        xs, ys = zip(*features_list)
        plt.scatter(xs, ys, marker='x', color='r')

        plt.axis("square")
        plt.pause(0.01)

if __name__ == '__main__':
    # import cProfile
    # cProfile.run("main()", sort="cumtime")
    main()
