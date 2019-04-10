import time

import numpy as np


def exponential_similarity(x, mu, v):
    diff = x - mu
    return np.exp(-0.5 * np.sum(diff*diff, axis=1) / v)


class ParticleFilter(object):
    # noise vector: speed, heading, perception location,
    #               perception false positive, perception false negative
    def __init__(self, n_particles, noise_vec, track, camera, collision_radius,
                 target_latency=None):
        self.target_latency = target_latency
        self.measured_latency = target_latency

        self.n_particles = n_particles
        self.track = track
        self.camera = camera
        self.collision_radius = collision_radius

        self.speed_noise = noise_vec[0]
        self.heading_noise = noise_vec[1]
        self.perception_noise = noise_vec[2]
        self.false_pos_rate = noise_vec[3]
        self.false_neg_rate = noise_vec[4]

        self.last_speed = 0
        self.last_yaw_rate = 0

        self.particles = self.random_particles(self.n_particles)
        self.weights = np.zeros(self.n_particles)

    def get_num_particles(self):
        return self.n_particles

    def init_particles_position(self, x, y, h, x_noise, y_noise, h_noise):
        for i in range(self.n_particles):
            self.particles[:, i] = [x, y, h]
            self.particles[:, i] += np.random.normal(0, [x_noise, y_noise, h_noise])

    def random_particles(self, n_particles):
        w = self.track.x_max - self.track.x_min
        h = self.track.y_max - self.track.y_min
        return np.stack([
            self.track.x_min + w * np.random.random(n_particles),
            self.track.y_min + h * np.random.random(n_particles),
            2 * np.pi * np.random.random(n_particles)
        ])

    def stddev(self, top_pct=1.0):
        n = max(int(self.n_particles * top_pct), 1)
        best_particles_idxs = np.argsort(self.weights)[-n:]
        return np.std(self.particles[:, best_particles_idxs], axis=1)

    def mean(self, top_pct=1.0):
        n = max(int(self.n_particles * top_pct), 1)
        best_particles_idxs = np.argsort(self.weights)[-n:]
        best_particles = self.particles[:, best_particles_idxs]

        mx, my = np.mean(best_particles[:2], axis=1)
        unit_xs = np.cos(best_particles[2])
        unit_ys = np.sin(best_particles[2])
        mh = np.math.atan2(np.mean(unit_ys), np.mean(unit_xs))

        return np.array([mx, my, mh])

    def rectify_particles(self):
        # reset_mask = self.particles[0] < self.track.x_min
        # reset_mask = np.logical_or(reset_mask, self.particles[0] > self.track.x_max)
        # reset_mask = np.logical_or(reset_mask, self.particles[1] < self.track.y_min)
        # reset_mask = np.logical_or(reset_mask, self.particles[1] > self.track.y_max)
        reset_mask = self.track.collision_check(self.particles[:2], self.collision_radius)
        # reset_mask = np.logical_or(reset_mask, wall_check)

        n = np.count_nonzero(reset_mask)
        if n > 0:
            self.particles[:, reset_mask] = self.random_particles(n)

        self.particles[2] = self.particles[2] % (2 * np.pi)

    def predict(self, speed, yaw_rate, dt):
        self.rectify_particles()

        speed_noise_vec = np.random.normal(0, self.speed_noise, self.n_particles)
        path_dist = (speed + speed_noise_vec) * dt
        speed_probs = exponential_similarity(speed_noise_vec.reshape(-1,1),
                                             0, self.speed_noise)

        yaw_rate_noise_vec = np.random.normal(0, self.heading_noise, self.n_particles)
        dh = (yaw_rate + yaw_rate_noise_vec) * dt
        yaw_probs = exponential_similarity(yaw_rate_noise_vec.reshape(-1,1),
                                           0, self.heading_noise)

        h_travel = self.particles[2] + dh / 2

        dx = np.cos(h_travel) * path_dist
        dy = np.sin(h_travel) * path_dist

        self.particles += np.stack([dx, dy, dh])
        self.weights = speed_probs * yaw_probs

    def observe(self, obs):
        # perception: each row is (x, y) of observed red blob
        latency_start = time.time()

        prediction = self.camera.project_onto_image(self.particles, self.track.features)

        for i in xrange(self.n_particles):
            img_feats = prediction[i]

            if img_feats is None and obs is None:
                p = 1.0
            elif img_feats is None and obs is not None:
                p = self.false_pos_rate
            elif img_feats is not None and obs is None:
                p = self.false_neg_rate
            else:
                p = np.max(exponential_similarity(obs, img_feats, self.perception_noise))
                # print p

            self.weights[i] *= p

        self.measured_latency = time.time() - latency_start

        if self.target_latency is not None:
            gamma = 0.5
            grow_prop = gamma + (1-gamma) * self.target_latency / self.measured_latency
            self.n_particles = int(self.n_particles * grow_prop)
            if self.n_particles < 1:
                self.n_particles = 1

    def resample(self):
        normed_weights = self.weights / np.sum(self.weights)
        idxs = np.arange(len(self.weights))

        resample_idxs = np.random.choice(idxs, size=self.n_particles, p=normed_weights)

        self.particles = self.particles[:, resample_idxs]
        self.weights = np.zeros(self.n_particles)
