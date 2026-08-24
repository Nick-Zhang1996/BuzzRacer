from types import SimpleNamespace

import pytest

from buzzracer.controllers.controller import Controller
from buzzracer.controllers.joystick_controller import (
    JoystickController,
    JoystickControllerConfig,
    JoystickControllerState,
)


class _FakeJoystick:
    def __init__(self, axes):
        self._axes = axes

    def get_axis(self, index):
        return self._axes[index]

    def quit(self):
        return None


def test_joystick_controller_is_registered():
    assert Controller.registry['JoystickController'] is JoystickController


def test_joystick_controller_maps_stick_throttle_and_right_stick_steering(monkeypatch):
    monkeypatch.setattr(JoystickController, '_pump_events', staticmethod(lambda: None))

    config = JoystickControllerConfig(SimpleNamespace(dt=0.01), SimpleNamespace())
    state = JoystickControllerState(config)
    state.joystick = _FakeJoystick([0.0, -1.0, 0.0, -0.5])
    state.axis_count = 4

    car_params = SimpleNamespace(
        max_steer_left=0.4,
        max_steer_right=0.6,
        max_throttle=0.8,
        min_throttle=-1.0,
    )

    ctrl, valid, _, msg = JoystickController.control(
        car_state=SimpleNamespace(),
        car_params=car_params,
        track=None,
        controller_config=config,
        controller_state=state,
        main_state=SimpleNamespace(),
        car_index=0,
    )

    assert valid is True, msg
    assert ctrl.steering < 0.0
    assert ctrl.steering > -car_params.max_steer_right
    assert ctrl.throttle == pytest.approx(car_params.max_throttle)
