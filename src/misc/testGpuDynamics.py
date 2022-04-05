from DynamicSimulator import DynamicSimulator
from Car import Car
from math import radians
import numpy as np
class Dummy():
    def __init__(self):
        pass

car = Car.Factory(Dummy(), "porsche", controller=None,init_states=(3.7*0.6,3.75*0.6, radians(-90),2.0))
car.noise = False
car_state = np.array([2.4847, 1.9582, 0.1866, 0.9854, 0.0832, 1.6597])
sim_state = np.array([2.4847, 0.9529, 1.9582, 0.2645, 0.1866, 1.6597])
control = np.array([0.3974, 0.0356])

DynamicSimulator.dt = 0.01
DynamicSimulator.advanceDynamics(car_state, control, car)
