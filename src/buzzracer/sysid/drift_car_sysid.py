import numpy as np
import matplotlib.pyplot as plt

data = [-0.2, -40,
        -0.3, -90,
        -0.4, -145,
        -0.5, -191,
        -0.6, -225,
        -0.7, -258,
        -0.8, -276,
        -0.9, -315,
        -1.0, -339,
        0.2, 42,
        0.3, 95,
        0.4, 152,
        0.5, 193,
        0.6, 230,
        0.7, 267,
        0.8, 296,
        0.9, 323,
        1.0, 345]
data = np.asarray(data).reshape(-1, 2)
positive_data = data[data[:, 0] > 0]
wheel_speed_rad = positive_data[:, 1] * (2.0 * np.pi / 60.0)
throttle = positive_data[:, 0]

quadratic_fit = np.polyfit(wheel_speed_rad, throttle, 2)
wheel_speed_fit = np.linspace(wheel_speed_rad.min(), wheel_speed_rad.max(), 200)
throttle_fit = np.polyval(quadratic_fit, wheel_speed_fit)

print(
    'throttle = '
    f'{quadratic_fit[0]:.8f} * wheel_speed_rad^2 + '
    f'{quadratic_fit[1]:.8f} * wheel_speed_rad + '
    f'{quadratic_fit[2]:.8f}'
)

plt.plot(wheel_speed_rad, throttle, 'o')
plt.plot(wheel_speed_fit, throttle_fit, '-')
plt.xlabel('wheel speed (rad/s)')
plt.ylabel('throttle')
plt.show()
