from extension.Extension import Extension

import matplotlib.patches
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import numpy as np
from scipy import stats
from math import pi, sin, cos, tan, atan
from sysid.tire import tireCurve


# hopefully not necessary
class ParticleFilter(Extension):
    """This class is essentially a wrapper for the particle_filter class so that it can
    play nice within the greater context of the overall simulation."""
    def __init__(self, main):
        super().__init__(main)
        self.numParticles = main.params['numberOfParticles']
        self.minEffectiveNumParticles = self.numParticles / 8

    def init(self):
        self.particle_filter = particle_filter(self.numParticles, self.main.track, init_state=self.main.cars[0].states,
                                               uniform=False)

    def preUpdate(self):
        # move particles through dynamics model
        # do this before car.states is actually updated
        # todo: share data/eliminate this so that the whole controller -> dynamics process isn't done twice
        # once here and once for the simulator
        car = self.main.cars[0]
        throttle, steering, _, _ = car.controller.ctrlCar(car.states, self.main.track)
        newStates = self.main.simulator.advanceDynamics(car.states, (throttle, steering), car)  # output as np array
        stateChange = newStates - car.states
        stateChange = stateChange[0:3]
        oldHeading = car.states[2]
        self.particle_filter.advanceParticles(stateChange, oldHeading)

    def update(self):
        # take measurements from car and evaluate likelihoods of observation for each particle
        measurements = self.main.track.simulateSensorsOnlyNums(self.main.cars[0].states)
        measurements = measurements.reshape(1, 4)
        pf = self.particle_filter
        measurements += pf.addMeasurementNoise()

        # this takes **forever**
        pf.findPosteriors(measurements, self.main.track)
        pf.resampleIfNeeded()

        pf.calcBestEstimate(method='average') #'max'

class DiscreteDistribution:
    def __init__(self, numParticles, array=None):
        self.numParticles = numParticles
        if not array:
            self.particles = np.ones((numParticles, 4))  # a row (particle) is: x, y, heading, weight
        else:
            self.particles = np.copy(array)

    def copy(self):
        return DiscreteDistribution(self.numParticles, self.particles)

    # def argMax(self):
    #     """
    #     Return the key with the highest value.
    #     """
    #     if len(self.particles) == 0:
    #         return None
    #     all = list(self.items())
    #     values = [x[1] for x in all]
    #     maxIndex = values.index(max(values))
    #     return all[maxIndex][0]

    def total(self):
        """
        Return the sum of values for all keys.
        """
        return float(sum(self.particles[:, 3]))

    def normalize(self):
        tot = self.total()
        if tot == 0.0:
            return
        self.particles[:, 3] /= tot

    def sample(self):
        self.normalize()

        x = random.random()

        for index in range(self.numParticles):
            if x <= self.particles[index, 3]:
                return self.particles[index, 0:3]  # top bound is exclusive -> cols 0 thru 2
            x -= self.particles[index, 3]


class particle_filter(DiscreteDistribution):
    def __init__(self, numParticles, track, init_state=None, uniform=False, array=None):
        if array:
            super().__init__(numParticles, array)
        else:
            super().__init__(numParticles)

        # determine (simplistic - no inner borders, no rounded corners) bounding rectangle of track
        self.xMin = 0
        self.xMax = track.gridsize[1] * track.scale
        self.yMin = 0
        self.yMax = track.gridsize[0] * track.scale

        # set minimum number of effective particles
        self.minEffectiveNumParticles = self.numParticles / 4

        # Distribution parameters
        # Process noise
        self.procNoiMean = (0, 0, 0)  # x, y, heading
        self.procNoiDistSD = 0.005 #0.03
        self.procNoiAngSD = 0.01 #0.14
        self.procNoiCov = [[self.procNoiDistSD ** 2, 0, 0],
                           [0, self.procNoiDistSD ** 2, 0],
                           [0, 0, self.procNoiAngSD ** 2]]

        # Measurement noise
        # self.measNoiMean = (0, 0)
        # self.measNoiSD = (0.01, 0.01)
        # self.measNoiCov = [[self.measNoiSD[0] ** 2, 0],
        #                    [0, self.measNoiSD[1] ** 2]]
        self.measNoiMean = (0, 0, 0, 0)
        self.measNoiSD = (0.01, 0.01, 0.01, 0.01)
        self.measNoiCov = [[self.measNoiSD[0] ** 2, 0, 0, 0],
                           [0, self.measNoiSD[1] ** 2, 0, 0],
                           [0, 0, self.measNoiSD[1] ** 2, 0],
                           [0, 0, 0, self.measNoiSD[1] ** 2]]

        if init_state is not None and uniform == False:
            # Starting distribution. Set up for 2D Gaussian starting distribution around starting point
            self.startMean = (init_state[0], init_state[1], init_state[2])
            self.startDistStdDev = 0.03  # 3 cm std dev
            self.startAngStdDev = 0.14  # 0.14 rad std dev
            # diagonal, no interrelation
            self.startCov = [[self.startDistStdDev ** 2, 0, 0],
                             [0, self.startDistStdDev ** 2, 0],
                             [0, 0, self.startAngStdDev ** 2]]
            self.initializeGaussian()
        else:
            # says reinitialize, but really its just initializing as uniform
            # called reinitialize because you might need to do this if you diverge from the true state
            self.reinitializeDistribution()

        self.bestEstimate = None
        self.calcBestEstimate()

    def initializeGaussian(self):
        self.particles[:, 0:3] = np.random.multivariate_normal(self.startMean, self.startCov, self.numParticles)
        self.normalize()

    def reinitializeDistribution(self):
        """If all weights go to zero, reinitialize the distribution to be uniform over the entire track."""
        # todo: implement checks for being in bounds
        # self.particles[:, 0:3] = np.random.uniform([0, 0.25, -pi], [2.5, 3.75, pi], (self.numParticles, 3))
        self.particles[:, 0:3] = np.random.uniform([self.xMin, self.yMin, -pi], [self.xMax, self.yMax, pi],
                                                   (self.numParticles, 3))
        self.particles[:, 3] = 1
        self.normalize()
        return

    def advanceParticles(self, masterStateChange, masterOldHeading):
        transform = np.zeros((self.numParticles * 3, 3))
        headingDifferences = self.particles[:, 2] - masterOldHeading
        transform[range(0, 3 * self.numParticles, 3), 0] = np.cos(headingDifferences)
        transform[range(1, 3 * self.numParticles, 3), 0] = np.sin(headingDifferences)
        transform[range(0, 3 * self.numParticles, 3), 1] = -np.sin(headingDifferences)
        transform[range(1, 3 * self.numParticles, 3), 1] = np.cos(headingDifferences)
        transform[range(2, 3 * self.numParticles, 3), 2] = 1
        particleChanges = transform @ masterStateChange
        particleChanges = particleChanges.reshape(self.numParticles, 3)
        self.particles[:, 0:3] += particleChanges
        self.addProcessNoise()
        return

    def addProcessNoise(self):
        processNoise = np.random.multivariate_normal(self.procNoiMean, self.procNoiCov, self.numParticles)
        self.particles[:, 0:3] += processNoise
        return

    def findPosteriors(self, measurements, track):
        measurements = np.array(measurements)
        theorMeass = np.zeros((self.numParticles, 4))

        # todo: generalize this to full track (what is heading "supposed" to be?)
        mask = np.ones((self.numParticles, ))  # for now, assuming car is going the right way
        # mask = np.logical_and(-pi < self.particles[:, 2], self.particles[:, 2] < 0) # correct for down straight
        facingRightWayInds = mask.nonzero()

        rawTheorMeas = np.zeros((self.numParticles, 4))
        i = 0
        for particle in self.particles:
            # todo: this is absolutely terrible efficiency wise. O(n^2)
            rawTheorMeas[i, :] = track.simulateSensorsOnlyNums(particle)
            i += 1
        theorMeass[facingRightWayInds, :] = rawTheorMeas[facingRightWayInds, :]

        mask = np.logical_not(mask)
        facingWrongWayInds = mask.nonzero()
        theorMeass[facingWrongWayInds, :] = rawTheorMeas[facingWrongWayInds, :]

        # print("sanity check:", all(sum(theorMeass, 1) >= 0.2))
        # theorMeass = np.hstack((np.reshape(2.5 - self.particles[:, 0], (self.numParticles, 1)),
        #                         np.reshape(self.particles[:, 0] - 2.0, (self.numParticles, 1))))

        likelihoods = self.evalSensingLikelihood(theorMeass - measurements)
        combinedLikelihoods = np.ones((self.numParticles,))
        for ind in range(len(measurements)):
            combinedLikelihoods *= likelihoods[:, ind]
        # plt.plot(self.particles[:, 0], likelihoods, '.')
        # plt.show()
        unnormedPosteriors = combinedLikelihoods * self.particles[:, 3]
        self.particles[:, 3] = np.reshape(unnormedPosteriors, (self.numParticles,))
        self.normalize()

    def evalSensingLikelihood(self, difference):
        sensingLikelihood = stats.norm.pdf(np.reshape(difference, (self.numParticles, 4)),
                                           self.measNoiMean, self.measNoiSD)
        return sensingLikelihood

    def resampleIfNeeded(self):
        effectiveNumParticles = 1 / sum(self.particles[:, 3] ** 2)
        if effectiveNumParticles < self.minEffectiveNumParticles:
            print(f'Effective number of particles was {effectiveNumParticles}!')
            print(f'Resampling to prevent degeneracy!')
            self.resampleFromPosteriors()

    def resampleFromPosteriors(self):
        if self.total() == 0:
            self.reinitializeDistribution()
        else:
            self.particles[:, 0:3] = np.array([self.sample() for i in range(self.numParticles)])
            self.particles[:, 3] = 1
            self.normalize()

    def addMeasurementNoise(self):
        measNoise = np.random.multivariate_normal(self.measNoiMean, self.measNoiCov, 1)
        return measNoise

    def calcBestEstimate(self, method='max'):
        bestEstimate = None
        if method == 'average':
            bestEstimate = np.average(self.particles[:, 0:3], axis=0, weights=self.particles[:, 3])
        else:
            maxWeight = -999999
            for particle in self.particles:
                if particle[3] > maxWeight:
                    maxWeight = particle[3]
                    bestEstimate = particle
        self.bestEstimate = bestEstimate


def advanceDynamics(state, control):
    L = 0.09  # car.L
    lf = 0.04824  # car.lf
    lr = L - lf

    Iz = 417757e-9  # car.Iz
    m = 0.1667  # car.m
    dt = 0.01  # DynamicSimulator.dt

    # NOTE here vx = vf, vy = vs, different convention
    x, y, heading, vx, vy, omega = state
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

    state = x, y, heading, vx, vy, omega
    return np.array(state)


if __name__ == "__main__":
    numberOfParticles = 1024
    minEffetiveNumParticles = 64
    particles = ParticleFilter(numberOfParticles, starting_distribution="uniform")
    archive = particles.particles

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(archive[:, 0], archive[:, 1], archive[:, 3])
    # plt.show()
    # plt.plot(particles.particles[:, 0], particles.particles[:, 1], '.')
    # plt.show()

    trueState = (2.25, 2.0, -pi / 2, 1, 0, 0)
    for t in range(1000):
        if t % 1 == 0:
            if not t == 0:
                archive = np.hstack((archive, particles.particles))
            _, cols = archive.shape
            minY = 999999
            maxY = -999999
            probMax = -999999
            if cols / 4 <= 4:
                low = 0
            else:
                low = int(cols / 4 - 4)

            fig, axes = plt.subplots(2, 1, figsize=(7, 10))
            for ind in range(low, int(cols / 4)):
                # get values to plot border lines
                tempMin = np.min(archive[:, ind * 4 + 1])
                tempYMax = np.max(archive[:, ind * 4 + 1])
                tempProbMax = np.max(archive[:, ind * 4 + 3])
                if tempMin < minY:
                    minY = tempMin
                if tempYMax > maxY:
                    maxY = tempYMax
                if tempProbMax > probMax:
                    probMax = tempProbMax

                # plot position data (x and y)
                axes[0].scatter(archive[:, ind * 4], archive[:, ind * 4 + 1], s=1000 * archive[:, ind * 4 + 3])
                # plot weight data (weight vs x)
                axes[1].plot(archive[:, ind * 4], archive[:, ind * 4 + 3], '.')
                # plt.subplot(313)
                # plt.plot(archive[:, ind * 4], archive[:, ind * 4 + 2], '.')
                # if ind == int(cols/4) - 1:
                #     fig = plt.figure(2)
                #     ax = fig.add_subplot(111, projection='3d')
                #     ax.scatter(archive[:, ind * 4], archive[:, ind * 4 + 1], archive[:, ind * 4 + 3])

            # add decorations
            # fig = plt.figure(1)
            # again on position (x and y) plot
            # fig, axes = plt.subplots(2, 1, figsize=(10, 5))
            ax = axes[0]
            trueX = trueState[0]
            trueY = trueState[1]
            trueHead = trueState[2]
            rec = matplotlib.patches.Rectangle((trueX - 0.02, trueY - 0.045), 0.04, 0.09, color='black')  # ,
            # angle=trueHead, rotation_point='center')
            ax.add_patch(rec)
            # ax.plot([2.0, 2.0], [minY, maxY], color="black")
            # ax.plot([2.5, 2.5], [minY, maxY], color="black")
            bottomlim = 1.5
            toplim = 2.25
            leftlim = 1
            rightlim = 3
            ax.plot([2.0, 2.0], [bottomlim, toplim], color="black")
            ax.plot([2.5, 2.5], [bottomlim, toplim], color="black")
            ax.set_xlim(left=leftlim, right=rightlim)
            ax.set_ylim(bottom=bottomlim, top=toplim)
            ax.set_xlabel('X position')
            ax.set_ylabel('Y position')
            ax.set_title('Particle Locations')
            # plt.plot([trueX, trueX], [minY, maxY], color="blue")
            ax = axes[1]
            ax.plot([2.0, 2.0], [0, probMax], color="black")
            ax.plot([2.5, 2.5], [0, probMax], color="black")
            ax.plot([trueX, trueX], [0, probMax], color="blue")
            ax.set_xlim(left=leftlim, right=rightlim)
            ax.set_xlabel('X position')
            ax.set_ylabel('Weight')
            ax.set_title('Particle Weight vs. Particle X position')
            plt.subplots_adjust(hspace=0.25)
            plt.show()
        print(t)
        control = (0.1, 0)  # 0.2*cos(10 * pi * t / 180))
        # todo: make full state update - would there then be covariance btwn say x and vx?
        # e.g. if vx + noise > predicted vx, then delta x should be greater than expected, not less
        particles.advanceParticles(trueState, control)  # including trueState to provide velocities
        trueState = advanceDynamics(trueState, control)
        # print(f'x: {trueState[0]}')
        # print(f'vy: {trueState[4]}')
        measurements = particles.simulateSensorMeasurements(trueState)
        # print(f'advPart numPart {len(particles.particles)}')
        # measurements = [0.1, 0.1]
        # measurements = [0.1, 0.1, 2, 2]
        # measurements = np.array([0.1 - 0.1 * sin(pi * t / 180), 0.1 + 0.1 * sin(pi * t / 180)])
        particles.findPosteriors(measurements)
        # print(f'findPost numPart {len(particles.particles)}')
        # print(f'resamp numPart {len(particles)}')

        effectiveNumParticles = 1 / sum(particles.particles[:, 3] ** 2)
        if effectiveNumParticles < minEffetiveNumParticles:
            print(f'Effective number of particles was {effectiveNumParticles}!')
            print(f'Resampling to prevent degeneracy!')
            particles.resampleFromPosteriors()
