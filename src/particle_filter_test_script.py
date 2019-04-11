import time
import os

import matplotlib.pyplot as plt
import numpy as np

from particle_filter import CameraModel, Track, ParticleFilter


load_track = True
my_dir = os.path.dirname(__file__)


def main():
    # features_list = [(-2, 0), (-2,2), (2,-2)] + [(2,2)]*100
    # track = Track(features_list)
    if load_track:
        track = Track.load_file(os.path.join(my_dir, "particle_filter/track_mk111.pkl"))
    else:
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

    if load_track:
        truth.init_particles_position(0, 0, 0, 0, 0, 0)
        filt.init_particles_position(0, 0, 0, 0.1, 0.1, 0.1)
    else:
        truth.init_particles_position(0.25, -0.1, 0, 0, 0, 0)
        # filt.init_particles_position(0.5, 0, 0, 0.1, 0.1, 0.1)

    dt = 0.2
    for t in np.arange(0, 60.0, dt):
        if load_track:
            track_len = track.racing_line_timestep * (track.racing_line.shape[1] - 1)
            track_t = t % track_len
            x, y, h = truth.mean()
            dx = track.spline_x(track_t) - x
            dy = track.spline_y(track_t) - y
            dh = np.math.atan2(track.spline_y(track_t,1), track.spline_x(track_t,1)) - h
            dh = min([dh, dh+2*np.pi, dh-2*np.pi], key=abs)
            yaw_rate = dh / dt
            v = ((dx*dx + dy*dy) ** 0.5) / dt
        else:
            if t < 5:
                v = yaw_rate = 0
            else:
                v = 0.3
                yaw_rate = 0.5

        t0 = time.time()

        filt.predict(v, yaw_rate, dt)
        truth.predict(v, yaw_rate, dt)

        obs = camera.project_onto_image(truth.mean().reshape(3,1), track.features)
        obs = obs[0]
        if np.random.random() < 0.1:
            obs = None
        if obs is not None:
            obs = max(obs, key=lambda f: f[1])
            obs += np.random.normal(0, 10, size=(2,)).astype(np.int)

        filt.observe(obs)

        t1 = time.time()

        plt.clf()
        # xs, ys = zip(*features_list)
        # plt.scatter(xs, ys, marker='x', color='r')
        track.draw(show=False)

        plt.plot(filt.particles[0], filt.particles[1], 'b.', markersize=1.5)

        x, y, h = truth.mean()
        plt.arrow(x, y, 0.1*np.cos(h), 0.1*np.sin(h), width=0.03, color="orange", zorder=10)

        x, y, h = filt.mean(top_pct=0.5)
        std_x, std_y, _ = filt.stddev()
        is_converged = std_x < 0.05 and std_y < 0.05
        plt.arrow(x, y, 0.1*np.cos(h), 0.1*np.sin(h), width=0.025, zorder=11,
                  color="g" if is_converged else "r")

        t2 = time.time()
        filt.resample()
        t3 = time.time()

        print "time", np.round(t3 - t2 + t1 - t0, 4), \
              "n_particles", filt.get_num_particles()

        plt.axis("equal")
        plt.pause(0.01)

if __name__ == '__main__':
    # import cProfile
    # cProfile.run("main()", sort="cumtime")
    main()
