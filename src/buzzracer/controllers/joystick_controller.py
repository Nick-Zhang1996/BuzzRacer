"""Joystick-based manual controller using pygame."""
from __future__ import annotations

from typing import TYPE_CHECKING
import logging

import numpy as np

from buzzracer.common import LoggingFilter
from buzzracer.types import CartesianState, Control
from buzzracer.controllers.controller import Controller, ControllerConfig, ControllerState

try:
    import pygame
except ImportError as exc:
    pygame = None
    _PYGAME_IMPORT_ERROR = exc
else:
    _PYGAME_IMPORT_ERROR = None

if TYPE_CHECKING:
    from buzzracer.cars.car_param import CarParam
    from buzzracer.main import MainState, MainConfig
    from buzzracer.tracks.track import Track

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addFilter(LoggingFilter(interval=1.0))


class JoystickControllerConfig(ControllerConfig):
    """Read-only configuration for joystick axis mapping."""

    def __init__(self, main_config: MainConfig, car_param: CarParam):
        super().__init__(main_config, car_param)
        del car_param
        self.joystick_index = 0

        self.steering_axis = 3
        self.steering_invert = True
        self.steering_deadzone = 0.05
        self.steering_scale = 1.0

        self.throttle_axis = 1
        self.throttle_axis_is_trigger = False
        self.throttle_invert = True
        self.throttle_deadzone = 0.05
        self.throttle_scale = 1.0

        self.use_brake_axis = False
        self.brake_axis = 4
        self.brake_axis_is_trigger = True
        self.brake_invert = False
        self.brake_deadzone = 0.05
        self.brake_scale = 1.0
        self.max_speed = 1.0


class JoystickControllerState(ControllerState):
    """Mutable runtime state for the joystick device."""

    def __init__(self, config: JoystickControllerConfig):
        del config
        self.joystick = None
        self.joystick_name = None
        self.axis_count = 0
        self.pygame_initialized = False
        self.last_error = None
        self.v_override = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state['joystick'] = None
        state['pygame_initialized'] = False
        return state


@Controller.register(JoystickControllerConfig, JoystickControllerState)
class JoystickController(Controller):
    """Manual controller that maps joystick axes directly to car controls."""

    @staticmethod
    def _pump_events():
        if pygame is not None:
            pygame.event.pump()

    @staticmethod
    def _apply_deadzone(value: float, deadzone: float) -> float:
        deadzone = float(np.clip(deadzone, 0.0, 0.99))
        if abs(value) <= deadzone:
            return 0.0
        scaled = (abs(value) - deadzone) / (1.0 - deadzone)
        return float(np.copysign(scaled, value))

    @staticmethod
    def _apply_positive_deadzone(value: float, deadzone: float) -> float:
        deadzone = float(np.clip(deadzone, 0.0, 0.99))
        if value <= deadzone:
            return 0.0
        return float((value - deadzone) / (1.0 - deadzone))

    @staticmethod
    def _normalize_axis(raw_value: float,
                        invert: bool,
                        deadzone: float,
                        scale: float) -> float:
        value = -raw_value if invert else raw_value
        value = float(np.clip(value, -1.0, 1.0))
        value = JoystickController._apply_deadzone(value, deadzone)
        return float(np.clip(value * scale, -1.0, 1.0))

    @staticmethod
    def _normalize_trigger(raw_value: float,
                           invert: bool,
                           deadzone: float,
                           scale: float) -> float:
        value = float(np.clip(raw_value, -1.0, 1.0))
        value = (value + 1.0) * 0.5
        if invert:
            value = 1.0 - value
        value = JoystickController._apply_positive_deadzone(value, deadzone)
        return float(np.clip(value * scale, 0.0, 1.0))

    @staticmethod
    def _get_axis_value(controller_state: JoystickControllerState,
                        axis_index: int,
                        axis_name: str) -> float:
        if axis_index < 0:
            raise ValueError(f'{axis_name} axis must be >= 0, got {axis_index}')
        if controller_state.axis_count <= axis_index:
            raise ValueError(
                f'{axis_name} axis {axis_index} unavailable, joystick only has '
                f'{controller_state.axis_count} axes')
        return float(controller_state.joystick.get_axis(axis_index))

    @staticmethod
    def _ensure_joystick_ready(controller_config: JoystickControllerConfig,
                               controller_state: JoystickControllerState) -> tuple[bool, str]:
        if controller_state.joystick is not None:
            return True, 'Joystick ready'
        if pygame is None:
            return False, f'pygame is unavailable: {_PYGAME_IMPORT_ERROR}'

        try:
            if not controller_state.pygame_initialized:
                pygame.init()
                pygame.joystick.init()
                controller_state.pygame_initialized = True

            joystick_count = pygame.joystick.get_count()
            if joystick_count <= controller_config.joystick_index:
                return (
                    False,
                    f'Joystick index {controller_config.joystick_index} unavailable '
                    f'({joystick_count} connected)')

            joystick = pygame.joystick.Joystick(controller_config.joystick_index)
            joystick.init()
            controller_state.joystick = joystick
            controller_state.joystick_name = joystick.get_name()
            controller_state.axis_count = joystick.get_numaxes()
            controller_state.last_error = None
            logger.info(
                'Connected joystick %d: %s (%d axes)',
                controller_config.joystick_index,
                controller_state.joystick_name,
                controller_state.axis_count)
            return True, 'Joystick ready'
        except pygame.error as exc:
            controller_state.last_error = str(exc)
            return False, f'Unable to initialize joystick: {exc}'

    @staticmethod
    def _compute_throttle(controller_config: JoystickControllerConfig,
                          controller_state: JoystickControllerState,
                          car_params: CarParam) -> float:
        throttle_raw = JoystickController._get_axis_value(
            controller_state, controller_config.throttle_axis, 'Throttle')
        if controller_config.throttle_axis_is_trigger:
            throttle = JoystickController._normalize_trigger(
                throttle_raw,
                controller_config.throttle_invert,
                controller_config.throttle_deadzone,
                controller_config.throttle_scale)
        else:
            throttle = JoystickController._normalize_axis(
                throttle_raw,
                controller_config.throttle_invert,
                controller_config.throttle_deadzone,
                controller_config.throttle_scale)

        if not controller_config.use_brake_axis:
            return (
                throttle * car_params.max_throttle
                if throttle >= 0.0
                else -abs(throttle) * abs(car_params.min_throttle)
            )

        brake_raw = JoystickController._get_axis_value(
            controller_state, controller_config.brake_axis, 'Brake')
        if controller_config.brake_axis_is_trigger:
            brake = JoystickController._normalize_trigger(
                brake_raw,
                controller_config.brake_invert,
                controller_config.brake_deadzone,
                controller_config.brake_scale)
        else:
            brake = max(
                0.0,
                JoystickController._normalize_axis(
                    brake_raw,
                    controller_config.brake_invert,
                    controller_config.brake_deadzone,
                    controller_config.brake_scale))

        throttle_cmd = float(np.clip(throttle - brake, -1.0, 1.0))
        return (
            throttle_cmd * car_params.max_throttle
            if throttle_cmd >= 0.0
            else -abs(throttle_cmd) * abs(car_params.min_throttle)
        )

    def final(self):
        if self.state is None:
            return
        if self.state.joystick is not None:
            self.state.joystick.quit()
            self.state.joystick = None
            self.state.joystick_name = None
            self.state.axis_count = 0
        if pygame is not None and self.state.pygame_initialized:
            pygame.joystick.quit()
            self.state.pygame_initialized = False

    @staticmethod
    def control(car_state: CartesianState,
                car_params: CarParam,
                track: Track,
                controller_config: JoystickControllerConfig,
                controller_state: JoystickControllerState,
                main_state: MainState,
                car_index,
                planner_state=None):
        """Read joystick axes and map them to steering and throttle."""
        del car_state, track, main_state, car_index, planner_state

        ctrl = Control(steering=0.0, throttle=0.0)
        ready, msg = JoystickController._ensure_joystick_ready(
            controller_config, controller_state)
        if not ready:
            return (ctrl, False, controller_state, msg)

        try:
            JoystickController._pump_events()
            steering_raw = JoystickController._get_axis_value(
                controller_state, controller_config.steering_axis, 'Steering')
            steering_norm = JoystickController._normalize_axis(
                steering_raw,
                controller_config.steering_invert,
                controller_config.steering_deadzone,
                controller_config.steering_scale)
            steering = (
                steering_norm * car_params.max_steer_left
                if steering_norm >= 0.0
                else steering_norm * car_params.max_steer_right
            )

            throttle = JoystickController._compute_throttle(
                controller_config, controller_state, car_params)
            throttle = float(np.clip(throttle, car_params.min_throttle, car_params.max_throttle))

            ctrl = Control(steering=steering, throttle=throttle)
            controller_state.last_error = None
            return (ctrl, True, controller_state, 'Controller OK')
        except ValueError as exc:
            controller_state.last_error = str(exc)
            return (ctrl, False, controller_state, str(exc))
        except Exception as exc:
            if pygame is not None and isinstance(exc, pygame.error):
                controller_state.last_error = str(exc)
                controller_state.joystick = None
                controller_state.joystick_name = None
                controller_state.axis_count = 0
                return (ctrl, False, controller_state, str(exc))
            raise
