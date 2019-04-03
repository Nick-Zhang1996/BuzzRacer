import numpy as np


def exponential_similarity(x, mu, v):
    diff = x - mu
    return np.exp(-np.dot(diff, diff)/v)


class ParticleFilter(object):
    # noise vector: position, heading, perception location, 
    #               perception false positive, perception false negative
    def __init__(self, n_particles, noise_vec, track, camera):
        self.n_particles = n_particles
        self.track = track
        self.camera = camera

        self.pos_noise = noise_vec[0]
        self.heading_noise = noise_vec[1]
        self.perception_noise = noise_vec[2]
        self.false_pos_rate = noise_vec[3]
        self.false_neg_rate = noise_vec[4]

        self.last_speed = 0
        self.last_yaw_rate = 0

        self.particles = self.random_particles(self.n_particles)
        self.weights = np.ones(self.n_particles) / n_particles

    def random_particles(self, n_particles):
        w = self.track.x_max - self.track.x_min
        h = self.track.y_max - self.track.y_min
        return np.stack([
            self.track.x_min + w * np.random.random(n_particles),
            self.track.y_min + h * np.random.random(n_particles),
            2 * np.pi * np.random.random(n_particles)
        ])

    def stddev(self, top_pct):
        n = max(int(self.n_particles * top_pct), 1)
        best_particles_idxs = np.argsort(self.weights)[-n:]
        return np.std(self.particles[:, best_particles_idxs], axis=1)

    def mean(self, top_pct):
        n = max(int(self.n_particles * top_pct), 1)
        best_particles_idxs = np.argsort(self.weights)[-n:]
        best_particles = self.particles[:, best_particles_idxs]
        mx, my = np.mean(best_particles[:2], axis=1)
        unit_xs = np.cos(best_particles[2])
        unit_ys = np.sin(best_particles[2])
        mh = np.math.atan2(np.mean(unit_ys), np.mean(unit_xs))
        return np.array([mx, my, mh])

    def rectify_particles(self):
        reset_mask = self.particles[0] < self.track.x_min
        reset_mask = np.logical_or(reset_mask, self.particles[0] > self.track.x_max)
        reset_mask = np.logical_or(reset_mask, self.particles[1] < self.track.y_min)
        reset_mask = np.logical_or(reset_mask, self.particles[1] > self.track.y_max)

        n = np.count_nonzero(reset_mask)
        if n > 0:
            self.particles[:, reset_mask] = self.random_particles(n)

        self.particles[2] = self.particles[2] % (2 * np.pi)

    def predict(self, speed, yaw_rate, dt):
        dh = np.ones(self.n_particles) * (yaw_rate * dt)
        dx = np.cos(self.particles[2]) * (speed * dt)
        dy = np.sin(self.particles[2]) * (speed * dt)

        dx += np.random.normal(0, self.pos_noise * dt, size=self.n_particles)
        dy += np.random.normal(0, self.pos_noise * dt, size=self.n_particles)
        dh += np.random.normal(0, self.heading_noise * dt, size=self.n_particles)

        self.particles += np.stack([dx, dy, dh])

        self.rectify_particles()

    def update_weights(self, perception_features):
        # perception: each row is (x, y) of observed red blob
        for i in xrange(self.n_particles):
            # print "pose:", self.particles[:,i].round(2)[:2]

            pred_loc, inds = self.camera.project_onto_image(self.particles[:,i], 
                                                            self.track.features)
            # print "looking for features", inds

            if pred_loc is None:
                pred_loc = np.array([])

            unused_features = set(perception_features)
            p = 1.0
            for pred_loc in pred_loc[:2].T:
                x, y = pred_loc
                def similarity_metric(f):
                    return exponential_similarity(np.array(f), pred_loc, self.perception_noise)

                if len(unused_features) > 0:
                    most_similar = max(unused_features, key=similarity_metric)
                    partial_prob = similarity_metric(most_similar)
                else:
                    most_similar = None

                if most_similar is not None and partial_prob >= self.false_neg_rate:
                    # true match is more likely than false negative
                    p *= partial_prob
                    unused_features.remove(most_similar)
                else:
                    # more likely a false negative
                    p *= self.false_neg_rate

            # account for likelihood of unmatched features
            p *= (self.false_pos_rate ** len(unused_features))

            self.weights[i] = p
            # print p

    def resample(self):
        normed_weights = self.weights / np.sum(self.weights)
        idxs = np.arange(self.n_particles)
        n = int(self.n_particles * 1.0)
        particle_idxs = np.random.choice(idxs, size=n, p=normed_weights)

        self.particles = np.concatenate([
            self.particles[:, particle_idxs],
            self.random_particles(self.n_particles - n)
        ], axis=1)
