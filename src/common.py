import numpy as np
from enum import Enum, auto

class ExperimentType(Enum):
    Simulation = auto()
    Realworld = auto()

class PrintObject:
    def prefix(self):
        return "["+self.__class__.__name__+"]: "
    def print_error(self,*message):
        print('\033[91m',self.prefix(), 'ERROR ', *message, '\033[0m')
        raise RuntimeError
    def print_ok(self,*message):
        print('\033[92m',self.prefix(), *message, '\033[0m')

    def print_warning(self,*message):
        # yellow
        #print('\033[93m', *message, '\033[0m')
        # red
        print('\033[91m',self.prefix(), 'WARNING: ', *message, '\033[0m')

    def print_info(self,*message):
        print('\033[96m',self.prefix(), *message, '\033[0m')

# ----------

def print_error(*message):
    print('\033[91m', 'ERROR ', *message, '\033[0m')
    raise RuntimeError
def print_ok(*message):
    print('\033[92m', *message, '\033[0m')

def print_warning(*message):
    # yellow
    #print('\033[93m', *message, '\033[0m')
    # red
    print('\033[91m', 'WARNING: ', *message, '\033[0m')

def print_info(*message):
    print('\033[96m', *message, '\033[0m')

def ndarray(x):
    return np.asarray(x, dtype=np.float64)

def angular_difference(a, b):
    diff = a - b
    angle_diff = diff - np.floor(diff / (2 * np.pi)) * 2 * np.pi
    while angle_diff > np.pi:
        angle_diff -= 2 * np.pi
    while angle_diff < -np.pi:
        angle_diff += 2 * np.pi
    return angle_diff

class PrintObject:
    def __init__(self):
        #print_ok(self.prefix() + "in use")
        pass

    def prefix(self):
        return "["+self.__class__.__name__+"]: "

    def print_error(self, *message):
        print('\033[91m', self.prefix(),'ERROR ', *message, '\033[0m')
        raise RuntimeError

    def print_ok(self, *message):
        print('\033[92m',self.prefix(), *message, '\033[0m')

    def print_warning(self, *message):
        # yellow
        #print('\033[93m',self.prefix(), *message, '\033[0m')
        # red
        print('\033[91m',self.prefix(), 'WARNING: ', *message, '\033[0m')

    def print_info(self, *message):
        print('\033[96m',self.prefix(), *message, '\033[0m')

