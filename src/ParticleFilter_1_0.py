# patchwork of methods related to a particle filter
# meant to be a very simple proof of concept

import math
import numpy as np
from math import sin, cos, tan, atan, pi
from scipy import stats
import matplotlib.pyplot as plt
from track.RCPTrack import RCPTrack
from sysid.tire import tireCurve


def instantiateParticleDistribution(x, y, heading):
    numParticles = 4096
    particles = np.zeros((numParticles, 4))
    mean = (x, y, heading)
    distStdDev = 0.03  # 3 cm std dev
    angStdDev = 0.14
    cov = [[distStdDev ** 2, 0, 0], [0, distStdDev ** 2, 0], [0, 0, angStdDev ** 2]]
    particles[:, 0:3] = np.random.multivariate_normal(mean, cov, numParticles)
    particles[:, 3] = 1 / numParticles
    return particles


def advanceParticles(state, control, particles):
    x, y, heading, vx, vy, omega = state
    numParticles, _ = particles.shape
    for index in range(numParticles):
        state = particles[index, 0], particles[index, 1], heading, vx, vy, omega
        newState = advanceDynamics(state, control)
        particles[index, 0:2] = newState[0:2]
    particles[:, 0:3] += addProcessNoise(numParticles)
    return particles


def advanceDynamics(car_states, control):
    L = 0.09  # car.L
    lf = 0.04824  # car.lf
    lr = L - lf

    Iz = 417757e-9  # car.Iz
    m = 0.1667  # car.m
    dt = 0.01  # DynamicSimulator.dt

    # NOTE here vx = vf, vy = vs, different convention
    x, y, heading, vx, vy, omega = car_states
    throttle, steering = control

    # for small longitudinal velocity use kinematic model
    if (vx < 0.05):
        beta = atan(lr / L * tan(steering))
        norm = lambda a, b: (a ** 2 + b ** 2) ** 0.5
        # motor model
        d_vx = 6.17 * (throttle - vx / 15.2 - 0.333)
        vx = vx + d_vx * dt
        vy = norm(vx, vy) * sin(beta)
        d_omega = 0.0
        omega = vx / L * tan(steering)

        slip_f = 0
        slip_r = 0
        Ffy = 0
        Fry = 0

    else:
        slip_f = -np.arctan((omega * lf + vy) / vx) + steering
        slip_r = np.arctan((omega * lr - vy) / vx)

        # Ffy = Df * np.sin( C * np.arctan(B *slip_f)) * 9.8 * lr / (lr + lf) * m
        # Fry = Dr * np.sin( C * np.arctan(B *slip_r)) * 9.8 * lf / (lr + lf) * m
        Ffy = tireCurve(slip_f) * m * 9.8 * lr / (lr + lf)
        Fry = 1.15 * tireCurve(slip_r) * m * 9.8 * lf / (lr + lf)

        # Dynamics
        # d_vx = 1.0/m * (Frx - Ffy * np.sin( steering ) + m * vy * omega)
        d_vx = 6.17 * (throttle - vx / 15.2 - 0.333)
        d_vy = 1.0 / m * (Fry + Ffy * np.cos(steering) - m * vx * omega)
        d_omega = 1.0 / Iz * (Ffy * lf * np.cos(steering) - Fry * lr)

        # discretization
        vx = vx + d_vx * dt
        vy = vy + d_vy * dt
        omega = omega + d_omega * dt

        # back to global frame
    vxg = vx * cos(heading) - vy * sin(heading)
    vyg = vx * sin(heading) + vy * cos(heading)

    # update x,y, heading
    x += vxg * dt
    y += vyg * dt
    heading += omega * dt + 0.5 * d_omega * dt * dt

    car_states = x, y, heading, vx, vy, omega
    return np.array(car_states)


def addProcessNoise(numParticles):
    mean = 0
    distStdDev = 0.03  # these are the same as the instantiating Gaussian but no evidence for this to be true
    headStdDev = 0.14
    means = (mean, mean, mean)
    cov = [[distStdDev, 0, 0], [0, distStdDev, 0], [0, 0, headStdDev]]
    processNoise = np.random.multivariate_normal(means, cov, numParticles)
    return processNoise


def findPosteriors(measurements, particles):
    meanError = 0
    stdError = 0.03
    track = RCPTrack()
    numParticles, _ = particles.shape
    measurements = np.array(measurements)
    # ys = particles[:, 1]
    # headings = particles[:, 2]
    weights = np.reshape(particles[:, 3], (numParticles, 1))
    # theorLeftMeas, theorRightMeas = track.preciseTrackBoundary((x, y), heading)
    theorMeass = np.hstack((np.reshape(2.35 - particles[:, 0], (numParticles, 1)), np.reshape(particles[:, 0] - 2.15, (numParticles, 1))))
    likelihoods = np.ones((numParticles, 1))
    for ind in range(len(measurements)):
        likelihoods *= evalSensingLikelihood(theorMeass[:, ind] - measurements[ind])
    unnormedPosteriors = likelihoods * weights
    # print(f'like {likelihoods}')
    # print(f'weight {weights}')
    # print(unnormedPosteriors)
    particles[:, 3] = np.reshape(unnormedPosteriors, (numParticles,)) / np.sum(unnormedPosteriors)
    return particles


def evalSensingLikelihood(difference):
    meanError = 0
    stdError = 0.03
    sensingLikelihood = stats.norm.pdf(np.reshape(difference, (len(difference), 1)), meanError, stdError)
    return sensingLikelihood


def resampleFromPosteriors(particles):
    weights = particles[:, 3]
    numParticles = len(weights)
    weights *= len(weights)  # scale weights to sum to the number of particles
    weights = np.round(weights).astype(int)  # round to integers
    newParticles = np.zeros_like((sum(weights), 4))
    newInd = 0
    for index in range(numParticles):
        childParticles = np.ones((weights[index], 4))
        childParticles[:, 0:3] = particles[index, 0:3]
        # print(f'childParts {childParticles}')
        # print(f'newParts {newParticles}')
        newParticles[newInd:weights[index]+newInd, :] = childParticles
        newInd += weights[index]

    newParticles[:, 3] = 1 / len(newParticles)
    newParticles = newParticles[np.invert(newParticles[:, 0] == 0), :]
    return newParticles

if __name__ == "__main__":
    particles = instantiateParticleDistribution(2.25, 2.0, -math.pi/2)
    plt.plot(particles[:, 0], particles[:, 3], '.')

    for t in range(1000):
        state = (2.25, 2.0, -math.pi/2, 1, 0, 0)
        control = (0.1, 0)
        particles = advanceParticles(state, control, particles)
        print(f'advPart numPart {len(particles)}')
        measurements = np.array([0.1-0.1*sin(pi*t/180), 0.1+0.1*sin(pi*t/180)])
        particles = findPosteriors(measurements, particles)
        print(f'findPost numPart {len(particles)}')
        particles = resampleFromPosteriors(particles)
        # print(f'resamp numPart {len(particles)}')
        if t % 10 == 0:
            plt.plot(particles[:, 0], particles[:, 3], '.')
            plt.show()


