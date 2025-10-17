
from .Car import Car

class Orca(Car):
    def __init__(self,main):
        Car.__init__(self,main)
    def initParam(self):
        self.L = 0.09
        self.lf = 0.04824
        self.lr = self.L - self.lf

        self.width = self.params['width']
