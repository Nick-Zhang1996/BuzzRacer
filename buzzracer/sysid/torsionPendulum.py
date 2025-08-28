# measure rotational inertia using a torsion pendulum
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation
from time import sleep
from math import degrees
from buzzracer.extension.optitrack import _Optitrack
import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))


class TorsionPendulum:
    def __init__(self):
        self.op = _Optitrack(None)
        self.op.callback = self.callback_fun
        self.rotation = []

    def callback_fun(self, optitrack_id, position, rotation):
        assert (optitrack_id == 1000)
        x, y, z = position
        qx, qy, qz, qw = rotation
        r = Rotation.from_quat([qx, qy, qz, qw])
        rz, ry, rx = r.as_euler('ZYX', degrees=False)
        self.rotation.append([rz, ry, rx])

    def run(self):
        input('press enter to stop\n')
        self.op.quit()
        data = np.array(self.rotation)
        plt.plot(data[:, 0])
        plt.plot(data[:, 1])
        plt.plot(data[:, 2])
        plt.show()


if __name__ == '__main__':
    main = TorsionPendulum()
    main.run()
