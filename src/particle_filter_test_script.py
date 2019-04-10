import time

import matplotlib.pyplot as plt
import numpy as np

from particle_filter import CameraModel, Track, ParticleFilter


def main():
    # features_list = [(-2, 0), (-2,2), (2,-2)] + [(2,2)]*100
    # track = Track(features_list)
    track = Track.load('slslslsl', 0.5)

    camera = CameraModel(angle_down=0.1, height=0.05, fov_horizontal=np.radians(62.2),
                         img_width=640, img_height=480)

    pos_noise = 0.05
    yaw_noise = 0.1
    measurement_noise = 1000.0
    false_positive_rate = 0.01
    false_negative_rate = 0.01
    noise_vec = [pos_noise, yaw_noise, measurement_noise,
                 false_positive_rate, false_negative_rate]
    car_radius = 0.05

    truth = ParticleFilter(1, [0.01, 0.02, 0, 0, 0], track, camera, car_radius)
    filt = ParticleFilter(1000, noise_vec, track, camera, car_radius,
                          target_latency=0.01)

    truth.init_particles_position(0.25, -0.1, 0, 0, 0, 0)
    # filt.init_particles_position(0.5, 0, 0, 0.1, 0.1, 0.1)

    for t in xrange(100000000):
        t0 = time.time()

        if t < 30:
            filt.predict(0, 0, 0.1)
            truth.predict(0, 0, 0.1)
        else:
            filt.predict(0.3, 0.5, 0.1)
            truth.predict(0.3, 0.5, 0.1)

        obs = camera.project_onto_image(truth.mean().reshape(3,1), track.features)
        obs = obs[0]
        if np.random.random() < 0.1:
            obs = None
        if obs is not None:
            obs = max(obs, key=lambda f: f[1])
            obs += np.random.normal(0, 10, size=(2,)).astype(np.int)
        # obs = [tuple(f) for f in obs.T]
        # print "obs", obs
        filt.observe(obs)

        t1 = time.time()

        plt.clf()
        plt.plot(filt.particles[0], filt.particles[1], 'b.', markersize=1.5)

        x, y, h = truth.mean()
        plt.arrow(x, y, 0.1*np.cos(h), 0.1*np.sin(h), width=0.03, color="orange")

        x, y, h = filt.mean()
        std_x, std_y, _ = filt.stddev()
        is_converged = std_x < 0.05 and std_y < 0.05
        plt.arrow(x, y, 0.1*np.cos(h), 0.1*np.sin(h), width=0.025,
                  color="g" if is_converged else "r")

        t2 = time.time()
        filt.resample()

        print "time", np.round(time.time() - t2 + t1 - t0, 4), \
              "n_particles", filt.get_num_particles()

        # xs, ys = zip(*features_list)
        # plt.scatter(xs, ys, marker='x', color='r')
        track.draw(show=False)

        plt.axis("equal")
        plt.pause(0.01)

if __name__ == '__main__':
    # import cProfile
    # cProfile.run("main()", sort="cumtime")
    main()
