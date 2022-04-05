from common import ndarray, angular_difference
import numpy as np

class PidController:
    def __init__(self, p, i, d, dt, integral_limit, lpf_cutoff_freq):
        self.__p = p
        self.__i = i
        self.__d = d
        self.__integral = 0
        self.__last_error = 0
        self.__last_x0 = 0
        self.__dt = dt
        self.__integral_limit = integral_limit
        self.__alpha = (2 * np.pi * self.__dt * lpf_cutoff_freq) / (2 * np.pi * self.__dt * lpf_cutoff_freq + 1)

        # DEBUG
        self.mag_p = 0
        self.mag_i = 0
        self.mag_d = 0

    # dxdt: allow user to provide derivative externally, i.e. from a gyro
    def control(self, x0, x, dxdt=None):
        error = x0 - x
        # Low pass filter: larger lpf freq => error = error; small lpf freq => error = last error.
        error = (1 - self.__alpha) * self.__last_error + self.__alpha * error
        # P:
        p_term = self.__p * error
        # I:
        self.__integral += error * self.__dt
        self.__integral = np.clip(self.__integral, -self.__integral_limit, self.__integral_limit)
        i_term = self.__i * self.__integral
        # D:
        if dxdt is None:
            derivative = (error - self.__last_error) / self.__dt
        else:
            derivative = (x0 - self.__last_x0) / self.__dt - dxdt
        self.__last_x0 = x0
        d_term = self.__d * derivative
        # Update.
        self.__last_error = error
        self.mag_p = p_term
        self.mag_i = i_term
        self.mag_d = d_term
        return p_term + i_term + d_term

    def getDebug(self):
        return (self.mag_p, self.mag_i, self.mag_d)

