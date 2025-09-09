import os
from math import radians

import pytest
import numpy as np
import matplotlib.pyplot as plt

from buzzracer.types import CurvilinearState, CartesianState, Control
from buzzracer.scripts.run import Main
from buzzracer.common import BASEDIR, wrap
from buzzracer.sysid.kinematic_bicycle_model import KinematicBicycleModelCartesian, KinematicBicycleModelFrenet
from buzzracer.sysid.dynamic_bicycle_model import DynamicBicycleModelCartesian, DynamicBicycleModelFrenet


def get_dummy_main():
    ''' Build a dummy main with .cars and .track'''
    test_config_folder = os.path.join(
        BASEDIR, 'buzzracer', 'tests', 'test_configs')
    config_filename = os.path.join(
        test_config_folder, 'test_minimum.xml')
    if not os.path.exists(config_filename):
        raise FileNotFoundError
    main = Main(config_filename)
    return main


def are_points_collinear(points, tol=1e-3):
    """
    Check if 2D points lie (approximately) on the same line.

    Args:
        points: np.array of shape (N, 2)
        tol: tolerance for residual

    Returns:
        bool
    """
    if len(points) <= 2:
        return True  # any 2 points are collinear

    x = points[:, 0]
    y = points[:, 1]

    # Design matrix for y = a*x + b
    A = np.vstack([x, np.ones_like(x)]).T

    # Least squares fit
    sol, residuals, rank, s = np.linalg.lstsq(A, y, rcond=None)

    # residuals is sum of squared errors; but if exactly collinear, lstsq returns [].
    if residuals.size > 0:
        error = residuals[0]
    else:
        # Compute manually if lstsq didn't return residuals
        y_pred = A @ sol
        error = np.sum((y - y_pred) ** 2)

    print(f'colinear residual = {error}')
    return error <= tol


@pytest.mark.parametrize(
    'config_filename',
    [
        pytest.param(
            'test_kinematic_bicycle.xml', id='test_kinematic_bicycle'
        ),
        pytest.param(
            'test_dynamic_bicycle.xml', id='test_dynamic_bicycle'
        ),
    ],
)
def test_run_with_config(config_filename):
    config_filename = os.path.join(
        BASEDIR, 'buzzracer', 'tests', 'test_configs', config_filename)

    if not os.path.exists(config_filename):
        raise FileNotFoundError

    experiment = Main(config_filename)
    experiment.run()


def test_curv_to_from_cart():
    ''' Transform CurvilinearState to/from CartesianState'''
    np.random.seed(0)
    main = get_dummy_main()
    track = main.track

    for _ in range(10):
        # make random curv state
        s = np.random.uniform(0, track.raceline_len_m)
        n = np.random.uniform(-0.3, 0.3)
        rel_heading = np.random.uniform(radians(-30), radians(30))
        curv = CurvilinearState(progress=s,
                                lateral_err=n,
                                rel_heading=rel_heading,
                                v_forward=1.0,
                                v_sideway=0.0,
                                rel_omega=0.0)
        cart = track.curv_to_cart(curv)
        remake_curv = track.cart_to_curv(cart)
        print(f'curv {curv}')
        print(f'cart {cart}')
        print(f'remake_curv {remake_curv}')
        np.testing.assert_allclose(curv, remake_curv, atol=1e-5, rtol=1e-5)


def test_kinematic_bicycle_frenet():
    dynamics_model_class = DynamicBicycleModelFrenet
    main = get_dummy_main()
    car = main.cars[0]

    # In CurvilinearState we specify rel_omega w.r.t. ref curve
    # to create a car with zero angular velocity w.r.t. inertial frame
    # we must calculate the appropriate rel_omega
    def get_ref_omega(state, curvature):
        dsdt = (state.v_forward * np.cos(state.rel_heading)
                - state.v_sideway * np.sin(state.rel_heading)
                ) / (1-state.lateral_err*curvature)
        # Reference angular velocity
        omega_ref = dsdt * curvature
        return omega_ref*0

    np.random.seed(4)
    state_temp = CurvilinearState(progress=np.random.uniform(0, main.track.raceline_len_m),
                                  lateral_err=np.random.uniform(-0.1, 0.1),
                                  rel_heading=np.random.uniform(-0.1, 0.1),
                                  v_forward=1.0,
                                  v_sideway=0.0,
                                  rel_omega=0.0
                                  )
    state0 = CurvilinearState(progress=state_temp.progress,
                              lateral_err=state_temp.lateral_err,
                              rel_heading=state_temp.rel_heading,
                              v_forward=1.0,
                              v_sideway=0.0,
                              rel_omega=-get_ref_omega(state_temp,
                                                       main.track.curvature_s(state_temp.progress))
                              )
    # If no control, we should be travelling in a straight line
    control = Control(steering=0, throttle=1.0)
    state = state0
    state_vec = [state]
    for _ in range(40):
        state = state_vec[-1]
        state_vec.append(
            dynamics_model_class.advance_dynamics(
                state, control, car, 0.01, main.track.curvature_s(state.progress))
        )

    cart_state_vec = [main.track.curv_to_cart(curv) for curv in state_vec]
    points = np.array([[val.x, val.y] for val in cart_state_vec])
    visualize(main.track, points,
              msg='Visually check the car is driving s traight line, starting from *')
    # plt.plot([val.rel_heading for val in state_vec])
    # plt.show()
    return

    assert are_points_collinear(points)
    # If positive steering, we should be going left
    control = Control(steering=radians(15), throttle=1.0)
    state = state0
    state_vec = [state]
    for _ in range(40):
        state = state_vec[-1]
        state_vec.append(
            dynamics_model_class.advance_dynamics(
                state, control, car, 0.01, main.track.curvature_s(state.progress))
        )

    cart_state_vec = [main.track.curv_to_cart(curv) for curv in state_vec]
    points = np.array([[val.x, val.y] for val in cart_state_vec])
    assert wrap(cart_state_vec[-1].heading - cart_state_vec[0].heading) > 0
    visualize(main.track, points,
              msg='Visually check the car is turning a smooth left curve, starting from *')


def visualize(track, points, msg=''):
    ''' Plot track raceline and points (traj)
    Args:
        track: Track object
        points: np.ndarray (N, 2)
    '''
    plt.plot(track.raceline_points[0],
             track.raceline_points[1])
    plt.plot(points[:, 0], points[:, 1])
    plt.plot(points[0, 0], points[0, 1], '*')
    plt.legend()
    plt.title(msg)
    plt.axis('equal')
    plt.show()
