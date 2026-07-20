"""Small equilibrium checks for the drift drivetrain model."""
from math import pi

import matplotlib.pyplot as plt
import numpy as np

from buzzracer.cars.car_param import CarConfig
from buzzracer.sysid.drift_model import DriftModel


def bisect_root(residual, lower=0.0, upper=100.0):
    """Return the positive wheel-speed root of a force-balance residual."""
    lower_residual = residual(lower)
    upper_residual = residual(upper)
    if lower_residual * upper_residual > 0.0:
        raise ValueError('wheel-speed root is not bracketed')

    for _ in range(50):
        middle = 0.5 * (lower + upper)
        middle_residual = residual(middle)
        if lower_residual * middle_residual <= 0.0:
            upper = middle
        else:
            lower = middle
            lower_residual = middle_residual
    return 0.5 * (lower + upper)


def rpm(wheel_speed):
    return wheel_speed * 60.0 / (2.0 * pi)


def constrained_check(car_param, throttle):
    """Return equilibrium wheel speed and total drive thrust at zero speed."""
    wheel_speed = bisect_root(
        lambda speed: DriftModel._drivetrain_force_balance_residual(
            speed, throttle, 0.0, 0.0, 0.0, 0.0, car_param))
    thrust = DriftModel.get_motor_fx(car_param, throttle, wheel_speed)
    return wheel_speed, thrust


def plot_tire_forces(car_param):
    """Plot the longitudinal arctan tire-force curves for both axles."""
    slip_ratio = np.linspace(-1.0, 1.0, 500)
    front_force = [DriftModel._combined_slip_force(
        slip, 0.0, car_param.drift_front_tire_A,
        car_param.drift_front_tire_B)[0] for slip in slip_ratio]
    rear_force = [DriftModel._combined_slip_force(
        slip, 0.0, car_param.drift_rear_tire_A,
        car_param.drift_rear_tire_B)[0] for slip in slip_ratio]

    plt.plot(slip_ratio, front_force,
             label=f'front (A={car_param.drift_front_tire_A}, B={car_param.drift_front_tire_B})')
    plt.plot(slip_ratio, rear_force,
             label=f'rear (A={car_param.drift_rear_tire_A}, B={car_param.drift_rear_tire_B})')
    plt.xlabel('slip ratio')
    plt.ylabel('longitudinal force (N)')
    plt.title('Drift tire force saturation')
    plt.grid()
    plt.legend()


if __name__ == '__main__':
    car_param = CarConfig.drift_rx7_30.value
    wheel_radius = car_param.drift_wheel_radius

    # Full throttle, no load.  The existing calibration measured 345 RPM.
    no_load_wheel_speed = bisect_root(
        lambda speed: DriftModel.get_motor_fx(car_param, 1.0, speed)
    )
    print('full throttle, no load:')
    print(f'  model:      {rpm(no_load_wheel_speed):.1f} RPM')
    print(f'  experiment: 345.0 RPM')

    # Constrain vx to zero and compare the wheel speed and forward thrust.
    for throttle in (0.25, 0.3, 0.4, 0.5, 0.7, 1.0):
        wheel_speed, thrust = constrained_check(car_param, throttle)
        print(f'vx=0, throttle={throttle:.2f}:')
        print(f'  wheel speed: {rpm(wheel_speed):.1f} RPM')
        print(f'  forward thrust: {thrust:.3f} N')

    plot_tire_forces(car_param)
    plt.show()
