import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d as plt3d

from track.RCPTrack import RCPTrack
from scipy.interpolate import interp1d, splev
from scipy.signal import savgol_filter
import numpy as np
import math


def plotError():
    L = 0
    points = []
    curvatures = []
    track = RCPTrack()
    track.load()
    xmax = 249
    xStep = 4
    ymax = 359
    yStep = 4
    for x in range(0, xmax, xStep):
        x /= 100
        for y in range(0, ymax, yStep):
            y /= 100
            state = (x, y, 0, 0, 0, 0)
            try:
                refPoint, n, refHeading, curvature, _, u = track.localTrajectory(state, L, True)
                s = track.uToS(u)
                u = track.sToU(s)
                xRef, yRef = splev(u, track.raceline, der=0)
                der = np.array(splev(u, track.raceline, der=1))
                headingRef = math.atan2(der[1], der[0])
                newx = xRef - n * np.sin(headingRef)
                newy = yRef + n * np.cos(headingRef)
                error = ((x - newx) ** 2 + (y - newy) ** 2) ** 0.5
                if abs(n) < 0.2:  # or error > 0.1
                    points.append([x, y, newx, newy, error])
                    curvatures.append([x, y, curvature])
            except:
                print("oops")

    plt.figure(1)
    for i in range(len(points)):
        plt.plot([points[i][0], points[i][2]], [points[i][1], points[i][3]])
    plt.show()

    xSize = xStep / 100
    ySize = yStep / 100
    bottom = np.zeros_like(points[:][0])

    plt.figure(2)
    ax = plt.axes(projection='3d')
    for i in range(0, len(points), 5):
        ax.bar3d(points[i][0], points[i][1], bottom, xSize, ySize, points[i][4])
    plt.show()
    plt.figure(3)
    ax1 = plt.axes(projection='3d')
    bottom = np.zeros_like(curvatures[:][0])
    for i in range(0, len(curvatures)):
        ax1.bar3d(curvatures[i][0], curvatures[i][1], bottom, xSize, ySize, curvatures[i][2])
    plt.show()


def plotCurvature():
    track = RCPTrack()
    track.load()
    discretized_raceline_len = 1024
    track.reconstructRaceline()
    ss = np.linspace(0, track.raceline_len_m, discretized_raceline_len)
    # rr = splev(ss % track.raceline_len_m, track.raceline_s, der=0)
    drr = np.array(splev(ss % track.raceline_len_m, track.raceline_s, der=1))

    _norm = lambda x: np.linalg.norm(x, axis=0)
    ddrr = vec_curvature = np.array(splev(ss % track.raceline_len_m, track.raceline_s, der=2))
    curv = 1.0 / (_norm(drr) ** 3 / (_norm(drr) ** 2 * _norm(ddrr) ** 2 - np.sum(drr * ddrr, axis=0) ** 2) ** 0.5)
    cross_curvature = drr[0] * vec_curvature[1] - drr[1] * vec_curvature[0]
    k_signed = np.copysign(curv, cross_curvature)

    # filter curvature
    k_signed_smooth = savgol_filter(k_signed, 50, 2)

    plt.plot(ss, k_signed, label="raw curvature")
    plt.plot(ss, k_signed_smooth, label="smoothed curvature")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # plotError()
    plotCurvature()
