"""CasADi-based nonlinear MPC controller for curvilinear trajectory tracking."""
from __future__ import annotations

from typing import TYPE_CHECKING
import logging
import os

import casadi as ca
import numpy as np
from scipy.interpolate import splev

from buzzracer.common import LoggingFilter, wrap
from buzzracer.controllers.controller import Controller, ControllerConfig, ControllerState
from buzzracer.controllers.pid_controller import PidController
from buzzracer.sysid.dynamic_bicycle_model import DynamicBicycleModelFrenet
from buzzracer.sysid.kinematic_bicycle_model import KinematicBicycleModelFrenet
from buzzracer.types import CartesianState, CurvilinearState, Control

if TYPE_CHECKING:
    from buzzracer.cars.car_param import CarParam
    from buzzracer.main import MainConfig, MainState
    from buzzracer.tracks.curvilinear_track import CurvilinearTrack

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addFilter(LoggingFilter(interval=1.0))


def _scalar_splev(value: float, tck) -> float:
    """Evaluate a scalar spline and always return a Python float."""
    return float(np.asarray(splev(value, tck, der=0)).reshape(-1)[0])


def _clip(val: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, val))


def _unwrap_scalar_near(value: float, anchor: float, period: float) -> float:
    return anchor + ((value - anchor + 0.5 * period) % period) - 0.5 * period


def _unwrap_progress_sequence(progress: np.ndarray,
                              anchor: float,
                              period: float) -> np.ndarray:
    """Unwrap a wrapped progress sequence into a monotone future-looking one."""
    values = np.asarray(progress, dtype=float).copy()
    if values.size == 0:
        return values

    values[0] = _unwrap_scalar_near(values[0], anchor, period)
    for i in range(1, values.size):
        candidate = _unwrap_scalar_near(values[i], values[i - 1], period)
        if candidate < values[i - 1]:
            candidate += period
        values[i] = candidate
    return values


def _interp_or_hold(query_t: np.ndarray,
                    source_t: np.ndarray,
                    source_values: np.ndarray) -> np.ndarray:
    """Linear interpolation with edge-value hold for short trajectories."""
    if source_t.size == 0:
        return np.zeros_like(query_t)
    if source_t.size == 1:
        return np.full_like(query_t, source_values[0], dtype=float)
    return np.interp(query_t, source_t, source_values,
                     left=source_values[0], right=source_values[-1])


class MpcControllerConfig(ControllerConfig):
    """Read-only MPC configuration."""

    def __init__(self, main_config: MainConfig, car_param: CarParam):
        super().__init__(main_config, car_param)
        del car_param
        self.dt = main_config.dt
        self.horizon = 20
        self.max_speed = 1.5
        self.max_offset = 0.5
        self.ref_speed_scale = 1.0
        self.min_ref_speed = 0.4

        self.dynamics_model = DynamicBicycleModelFrenet
        self.dynamics_model_name = 'dynamic_bicycle_frenet'

        self.progress_weight = 4.0
        self.lateral_weight = 50.0
        self.heading_weight = 18.0
        self.speed_weight = 6.0
        self.sideway_weight = 0.10
        self.yaw_rate_weight = 0.05

        self.terminal_progress_weight = 8.0
        self.terminal_lateral_weight = 80.0
        self.terminal_heading_weight = 30.0
        self.terminal_speed_weight = 8.0
        self.terminal_sideway_weight = 0.20
        self.terminal_yaw_rate_weight = 0.10

        self.steering_weight = 0.05
        self.throttle_weight = 0.03
        self.steering_rate_weight = 1.5
        self.throttle_rate_weight = 0.15

        self.solver_name = 'ipopt'
        self.solver_max_iter = 40
        self.solver_print_level = 0
        self.acceptable_tol = 1e-4
        self.warm_start = True

        self.export_codegen = False
        self.codegen_dir = ''
        self.codegen_cpp = True

        self.fallback_lateral_gain = 5.0
        self.fallback_heading_gain = 2.0
        self.fallback_yaw_damping = 0.15

    def stage_state_weights(self) -> np.ndarray:
        return np.array([
            self.progress_weight,
            self.lateral_weight,
            self.heading_weight,
            self.speed_weight,
            self.sideway_weight,
            self.yaw_rate_weight,
        ], dtype=float)

    def terminal_state_weights(self) -> np.ndarray:
        return np.array([
            self.terminal_progress_weight,
            self.terminal_lateral_weight,
            self.terminal_heading_weight,
            self.terminal_speed_weight,
            self.terminal_sideway_weight,
            self.terminal_yaw_rate_weight,
        ], dtype=float)

    def control_weights(self) -> np.ndarray:
        return np.array([self.steering_weight, self.throttle_weight], dtype=float)

    def control_rate_weights(self) -> np.ndarray:
        return np.array([self.steering_rate_weight, self.throttle_rate_weight], dtype=float)


class MpcControllerState(ControllerState):
    """Mutable controller state preserved across MPC iterations."""

    def __init__(self, config: MpcControllerConfig):
        self.solver_bundle: _CasadiMpcBundle | None = None
        self.last_u: np.ndarray | None = None
        self.last_x: np.ndarray | None = None
        self.last_applied_u = np.zeros(2, dtype=float)
        self.v_override = None

        self.reference_curv_traj = np.zeros((6, 0), dtype=float)
        self.predicted_curv_traj = np.zeros((6, 0), dtype=float)
        self.predicted_cart_traj = np.zeros((6, 0), dtype=float)
        self.debug_dict = {}

        self.fallback_speed_pid = PidController(1.0, 0.1, 0.01, config.dt, 2, 10)


class _CasadiMpcBundle:
    """Owns the symbolic MPC model, solver, and optional codegen artifacts."""

    STATE_DIM = 6
    CONTROL_DIM = 2

    def __init__(self, config: MpcControllerConfig, car_param: CarParam):
        self.config = config
        self.car_param = car_param
        self.horizon = config.horizon
        self.state_dim = self.STATE_DIM
        self.control_dim = self.CONTROL_DIM
        self.name_prefix = 'mpc_' + ''.join(
            char if char.isalnum() else '_' for char in (car_param.name or 'vehicle'))
        self._dynamics_fun = self._build_dynamics_fun()
        self._build_solver()
        if config.export_codegen and config.codegen_dir:
            self.export_codegen(config.codegen_dir)

    def _get_dynamics_model(self):
        configured_model = getattr(self.config, 'dynamics_model', None)
        if configured_model is not None:
            model = configured_model if isinstance(configured_model, type) else configured_model.__class__
            if hasattr(model, 'advance_dynamics_casadi'):
                return model

        model_map = {
            'dynamic_bicycle_frenet': DynamicBicycleModelFrenet,
            'kinematic_bicycle_frenet': KinematicBicycleModelFrenet,
        }
        try:
            model = model_map[self.config.dynamics_model_name]
        except KeyError as exc:
            raise ValueError(
                f'Unsupported dynamics_model_name={self.config.dynamics_model_name!r}'
            ) from exc
        if not hasattr(model, 'advance_dynamics_casadi'):
            raise AttributeError(
                f'{model.__name__} must define advance_dynamics_casadi for MpcController'
            )
        return model

    def _build_dynamics_fun(self):
        model = self._get_dynamics_model()
        x = ca.SX.sym('x', self.state_dim)
        u = ca.SX.sym('u', self.control_dim)
        curvature = ca.SX.sym('curvature')
        x_next = model.advance_dynamics_casadi(
            x, u, self.car_param, self.config.dt, curvature)
        return ca.Function(f'{self.name_prefix}_discrete_dynamics', [x, u, curvature], [x_next])

    @property
    def parameter_dim(self) -> int:
        nx = self.state_dim
        nu = self.control_dim
        n = self.horizon
        return nx + nu + nx * (n + 1) + n + nx + nx + nu + nu

    def _build_solver(self):
        nx = self.state_dim
        nu = self.control_dim
        n = self.horizon

        x_var = ca.SX.sym('X', nx, n + 1)
        u_var = ca.SX.sym('U', nu, n)
        params = ca.SX.sym('P', self.parameter_dim)

        offset = 0
        x0 = params[offset:offset + nx]
        offset += nx
        u_prev = params[offset:offset + nu]
        offset += nu
        x_ref = ca.reshape(params[offset:offset + nx * (n + 1)], nx, n + 1)
        offset += nx * (n + 1)
        curvature = params[offset:offset + n]
        offset += n
        w_state = params[offset:offset + nx]
        offset += nx
        w_terminal = params[offset:offset + nx]
        offset += nx
        w_u = params[offset:offset + nu]
        offset += nu
        w_du = params[offset:offset + nu]
        offset += nu
        assert offset == self.parameter_dim

        constraints = [x_var[:, 0] - x0]
        objective = 0
        for k in range(n):
            err = x_var[:, k] - x_ref[:, k]
            objective += ca.dot(w_state * err, err)
            objective += ca.dot(w_u * u_var[:, k], u_var[:, k])

            du = u_var[:, k] - (u_prev if k == 0 else u_var[:, k - 1])
            objective += ca.dot(w_du * du, du)

            x_next = self._dynamics_fun(x_var[:, k], u_var[:, k], curvature[k])
            constraints.append(x_var[:, k + 1] - x_next)

        terminal_err = x_var[:, n] - x_ref[:, n]
        objective += ca.dot(w_terminal * terminal_err, terminal_err)

        z = ca.vertcat(ca.reshape(x_var, -1, 1), ca.reshape(u_var, -1, 1))
        g = ca.vertcat(*constraints)
        self.objective_fun = ca.Function(f'{self.name_prefix}_objective', [z, params], [objective])
        self.constraint_fun = ca.Function(f'{self.name_prefix}_constraints', [z, params], [g])
        self.objective_gradient_fun = ca.Function(
            f'{self.name_prefix}_objective_gradient', [z, params], [ca.gradient(objective, z)])
        self.constraint_jacobian_fun = ca.Function(
            f'{self.name_prefix}_constraint_jacobian', [z, params], [ca.jacobian(g, z)])
        lam = ca.SX.sym('lam', g.numel())
        lagrangian = objective + ca.dot(lam, g)
        hess = ca.hessian(lagrangian, z)[0]
        self.lagrangian_hessian_fun = ca.Function(
            f'{self.name_prefix}_lagrangian_hessian', [z, params, lam], [hess])

        nlp = {'x': z, 'f': objective, 'g': g, 'p': params}
        solver_options = {
            'expand': True,
            'print_time': False,
            'ipopt.print_level': int(self.config.solver_print_level),
            'ipopt.max_iter': int(self.config.solver_max_iter),
            'ipopt.tol': float(self.config.acceptable_tol),
            'ipopt.acceptable_tol': float(self.config.acceptable_tol),
            'ipopt.sb': 'yes',
            'ipopt.warm_start_init_point': 'yes',
        }
        self.solver = ca.nlpsol(f'{self.name_prefix}_solver',
                                self.config.solver_name,
                                nlp,
                                solver_options)

        self.lbg = np.zeros(g.numel(), dtype=float)
        self.ubg = np.zeros(g.numel(), dtype=float)
        self.lbx, self.ubx = self._make_variable_bounds()

    def _make_variable_bounds(self):
        nx = self.state_dim
        nu = self.control_dim
        n = self.horizon
        x_count = nx * (n + 1)
        z_count = x_count + nu * n

        lbx = np.full(z_count, -np.inf, dtype=float)
        ubx = np.full(z_count, np.inf, dtype=float)

        v_index = np.arange(3, x_count, nx)
        vy_index = np.arange(4, x_count, nx)
        r_index = np.arange(5, x_count, nx)
        lbx[v_index] = 0.0
        ubx[v_index] = self.config.max_speed + 2.0
        lbx[vy_index] = -3.0
        ubx[vy_index] = 3.0
        lbx[r_index] = -30.0
        ubx[r_index] = 30.0

        steer_offset = x_count
        throttle_offset = x_count + 1
        for k in range(n):
            lbx[steer_offset + k * nu] = -self.car_param.max_steer_right
            ubx[steer_offset + k * nu] = self.car_param.max_steer_left
            lbx[throttle_offset + k * nu] = self.car_param.min_throttle
            ubx[throttle_offset + k * nu] = self.car_param.max_throttle

        return lbx, ubx

    def pack_parameters(self,
                        x0: np.ndarray,
                        u_prev: np.ndarray,
                        x_ref: np.ndarray,
                        curvature: np.ndarray) -> np.ndarray:
        return np.concatenate([
            np.asarray(x0, dtype=float).reshape(-1),
            np.asarray(u_prev, dtype=float).reshape(-1),
            np.asarray(x_ref, dtype=float).reshape(-1, order='F'),
            np.asarray(curvature, dtype=float).reshape(-1),
            self.config.stage_state_weights(),
            self.config.terminal_state_weights(),
            self.config.control_weights(),
            self.config.control_rate_weights(),
        ])

    def initial_guess(self,
                      x0: np.ndarray,
                      x_ref: np.ndarray,
                      last_x: np.ndarray | None,
                      last_u: np.ndarray | None) -> np.ndarray:
        nx = self.state_dim
        nu = self.control_dim
        n = self.horizon

        x_guess = np.array(x_ref, copy=True, dtype=float, order='F')
        x_guess[:, 0] = x0
        if last_x is not None and last_x.shape == (nx, n + 1):
            x_guess[:, :-1] = last_x[:, 1:]
            x_guess[:, -1] = last_x[:, -1]
            x_guess[:, 0] = x0

        u_guess = np.zeros((nu, n), dtype=float, order='F')
        if last_u is not None and last_u.shape == (nu, n):
            u_guess[:, :-1] = last_u[:, 1:]
            u_guess[:, -1] = last_u[:, -1]

        return np.concatenate([x_guess.reshape(-1, order='F'),
                               u_guess.reshape(-1, order='F')])

    def solve(self,
              x0: np.ndarray,
              u_prev: np.ndarray,
              x_ref: np.ndarray,
              curvature: np.ndarray,
              last_x: np.ndarray | None,
              last_u: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, dict]:
        params = self.pack_parameters(x0, u_prev, x_ref, curvature)
        z0 = self.initial_guess(x0, x_ref, last_x, last_u)
        solution = self.solver(x0=z0,
                               lbx=self.lbx,
                               ubx=self.ubx,
                               lbg=self.lbg,
                               ubg=self.ubg,
                               p=params)
        stats = self.solver.stats()
        z_opt = np.asarray(solution['x']).reshape(-1)

        nx = self.state_dim
        nu = self.control_dim
        n = self.horizon
        x_count = nx * (n + 1)
        x_opt = z_opt[:x_count].reshape((nx, n + 1), order='F')
        u_opt = z_opt[x_count:].reshape((nu, n), order='F')
        info = {
            'success': bool(stats.get('success', False)),
            'return_status': stats.get('return_status', ''),
            'iterations': stats.get('iter_count', None),
            'objective': float(np.asarray(solution['f']).reshape(-1)[0]),
        }
        return x_opt, u_opt, info

    def export_codegen(self, directory: str):
        """Emit reusable numerical kernels for offline compilation."""
        os.makedirs(directory, exist_ok=True)
        filename = os.path.join(directory, f'{self.name_prefix}_tracking_codegen.cpp' if self.config.codegen_cpp
                                else f'{self.name_prefix}_tracking_codegen.c')
        generator = ca.CodeGenerator(filename, {'cpp': self.config.codegen_cpp})
        generator.add(self._dynamics_fun)
        generator.add(self.objective_fun)
        generator.add(self.constraint_fun)
        generator.add(self.objective_gradient_fun)
        generator.add(self.constraint_jacobian_fun)
        generator.add(self.lagrangian_hessian_fun)
        generator.generate()


@Controller.register(MpcControllerConfig, MpcControllerState)
class MpcController(Controller):
    """Curvilinear nonlinear MPC for track and planner trajectory tracking."""

    def __init__(self):
        super().__init__()

    @staticmethod
    def _build_solver_bundle(controller_state: MpcControllerState,
                             controller_config: MpcControllerConfig,
                             car_params: CarParam):
        if controller_state.solver_bundle is None:
            controller_state.solver_bundle = _CasadiMpcBundle(controller_config, car_params)
        return controller_state.solver_bundle

    @staticmethod
    def _calc_throttle(state: CartesianState,
                       v_target: float,
                       car_params: CarParam,
                       throttle_pid: PidController) -> float:
        ss_throttle = car_params.ss_throttle_p0 * v_target + car_params.ss_throttle_p1
        ss_throttle = ss_throttle if v_target > 0 else 0.0
        throttle = throttle_pid.control(v_target, state.v_forward) + ss_throttle
        return _clip(throttle, car_params.min_throttle, car_params.max_throttle)

    @staticmethod
    def _fallback_control(car_state: CartesianState,
                          current_curv: np.ndarray,
                          ref_traj: np.ndarray,
                          car_params: CarParam,
                          controller_config: MpcControllerConfig,
                          controller_state: MpcControllerState) -> Control:
        ref0 = ref_traj[:, min(1, ref_traj.shape[1] - 1)]
        lateral_err = current_curv[1] - ref0[1]
        heading_err = wrap(current_curv[2] - ref0[2])
        steering = (-controller_config.fallback_lateral_gain * lateral_err
                    - controller_config.fallback_heading_gain * heading_err
                    - controller_config.fallback_yaw_damping * car_state.omega)
        steering = _clip(steering, -car_params.max_steer_right, car_params.max_steer_left)

        v_target = ref0[3]
        if controller_state.v_override is not None:
            v_target = min(v_target, controller_state.v_override)
        throttle = MpcController._calc_throttle(
            car_state, v_target, car_params, controller_state.fallback_speed_pid)
        return Control(steering=steering, throttle=throttle)

    @staticmethod
    def _track_reference_from_centerline(track: CurvilinearTrack,
                                         current_curv: np.ndarray,
                                         controller_config: MpcControllerConfig,
                                         controller_state: MpcControllerState) -> tuple[np.ndarray, np.ndarray]:
        horizon = controller_config.horizon
        dt = controller_config.dt
        track_len = track.data.raceline_len_m

        x_ref = np.zeros((6, horizon + 1), dtype=float, order='F')
        curvature = np.zeros(horizon, dtype=float)
        progress = float(current_curv[0])

        override_speed = controller_state.v_override
        for k in range(horizon + 1):
            progress_mod = progress % track_len
            target_speed = _scalar_splev(progress_mod, track.data.speed_s)
            target_speed *= controller_config.ref_speed_scale
            target_speed = min(target_speed, controller_config.max_speed)
            target_speed = max(target_speed, controller_config.min_ref_speed)
            if override_speed is not None:
                target_speed = min(target_speed, override_speed)

            x_ref[:, k] = np.array([progress, 0.0, 0.0, target_speed, 0.0, 0.0])
            if k < horizon:
                curvature[k] = _scalar_splev(progress_mod, track.data.curvature_s)
                progress += target_speed * dt
        return x_ref, curvature

    @staticmethod
    def _get_planner_cart_traj(planner_state, car_index: int) -> np.ndarray:
        with planner_state.traj_sync_lock:
            shape = (6, planner_state.car_count, planner_state.cart_traj_len.value)
            if shape[2] == 0:
                return np.zeros((6, 0), dtype=float)
            return np.frombuffer(planner_state.cart_traj_sync,
                                 dtype=np.float64,
                                 count=shape[0] * shape[1] * shape[2]
                                 ).reshape(shape, order='F')[:, car_index, :].copy()

    @staticmethod
    def _planner_reference(track: CurvilinearTrack,
                           current_curv: np.ndarray,
                           planner_state,
                           car_index: int,
                           controller_config: MpcControllerConfig,
                           controller_state: MpcControllerState) -> tuple[np.ndarray, np.ndarray] | None:
        cart_traj = MpcController._get_planner_cart_traj(planner_state, car_index)
        if cart_traj.shape[1] < 2:
            return None

        track_len = track.data.raceline_len_m
        curv_traj = np.zeros((6, cart_traj.shape[1]), dtype=float)
        for i in range(cart_traj.shape[1]):
            curv_traj[:, i] = track.cart_to_curv(CartesianState(*cart_traj[:, i])).to_tuple()

        curv_traj[0, :] = _unwrap_progress_sequence(curv_traj[0, :], current_curv[0], track_len)
        curv_traj[2, :] = np.unwrap(curv_traj[2, :])
        current_progress = _unwrap_scalar_near(current_curv[0], curv_traj[0, 0], track_len)

        start_idx = int(np.searchsorted(curv_traj[0, :], current_progress, side='left'))
        start_idx = min(max(start_idx, 0), curv_traj.shape[1] - 1)

        source_t = (np.arange(curv_traj.shape[1], dtype=float) - start_idx) * getattr(
            planner_state, 'traj_dt', controller_config.dt)
        query_t = np.arange(controller_config.horizon + 1, dtype=float) * controller_config.dt

        x_ref = np.zeros((6, controller_config.horizon + 1), dtype=float, order='F')
        for dim in range(6):
            x_ref[dim, :] = _interp_or_hold(query_t, source_t, curv_traj[dim, :])

        if controller_state.v_override is not None:
            x_ref[3, :] = np.minimum(x_ref[3, :], controller_state.v_override)
        x_ref[3, :] = np.minimum(x_ref[3, :], controller_config.max_speed)
        x_ref[3, :] = np.maximum(x_ref[3, :], 0.0)

        curvature = np.array([
            _scalar_splev(x_ref[0, k] % track_len, track.data.curvature_s)
            for k in range(controller_config.horizon)
        ], dtype=float)
        return x_ref, curvature

    @staticmethod
    def _curv_state_array(track: CurvilinearTrack,
                          car_state: CartesianState,
                          ref_progress_anchor: float | None = None) -> np.ndarray:
        curv = np.array(track.cart_to_curv(car_state).to_tuple(), dtype=float)
        if ref_progress_anchor is not None:
            curv[0] = _unwrap_scalar_near(curv[0], ref_progress_anchor, track.data.raceline_len_m)
        return curv

    @staticmethod
    def _curv_traj_to_cart(track: CurvilinearTrack, curv_traj: np.ndarray) -> np.ndarray:
        cart = np.zeros((6, curv_traj.shape[1]), dtype=float)
        for i in range(curv_traj.shape[1]):
            cart[:, i] = track.curv_to_cart(CurvilinearState(*curv_traj[:, i])).to_tuple()
        return cart

    @staticmethod
    def control(car_state: CartesianState,
                car_params: CarParam,
                track: CurvilinearTrack,
                controller_config: MpcControllerConfig,
                controller_state: MpcControllerState,
                main_state: MainState,
                car_index,
                planner_state=None):
        ctrl = Control(steering=0.0, throttle=0.0)

        if controller_config.planner:
            if planner_state is None or not planner_state.planner_ready.is_set():
                return (ctrl, False, controller_state, 'Planner not ready')

        bundle = MpcController._build_solver_bundle(controller_state, controller_config, car_params)

        ref_result = None
        if controller_config.planner and planner_state is not None:
            current_curv_mod = MpcController._curv_state_array(track, car_state)
            ref_result = MpcController._planner_reference(
                track, current_curv_mod, planner_state, car_index, controller_config, controller_state)
            if ref_result is None:
                logger.warning('Planner reference unavailable for %s, falling back to centerline',
                               car_params.name)

        if ref_result is None:
            current_curv = MpcController._curv_state_array(track, car_state)
            x_ref, curvature = MpcController._track_reference_from_centerline(
                track, current_curv, controller_config, controller_state)
        else:
            x_ref, curvature = ref_result
            current_curv = MpcController._curv_state_array(track, car_state, x_ref[0, 0])
        current_curv[2] = x_ref[2, 0] + wrap(current_curv[2] - x_ref[2, 0])

        if abs(current_curv[1] - x_ref[1, 0]) > controller_config.max_offset:
            msg = f'|lateral_error|={abs(current_curv[1] - x_ref[1, 0]):.3f} exceeds limit'
            return (ctrl, False, controller_state, msg)

        try:
            x_opt, u_opt, info = bundle.solve(
                x0=current_curv,
                u_prev=controller_state.last_applied_u,
                x_ref=x_ref,
                curvature=curvature,
                last_x=controller_state.last_x if controller_config.warm_start else None,
                last_u=controller_state.last_u if controller_config.warm_start else None,
            )
        except RuntimeError as exc:
            fallback_ctrl = MpcController._fallback_control(
                car_state, current_curv, x_ref, car_params, controller_config, controller_state)
            logger.warning('MPC solver exception for %s: %s', car_params.name, exc)
            return (fallback_ctrl, True, controller_state, 'MPC solver exception')

        if not info['success']:
            fallback_ctrl = MpcController._fallback_control(
                car_state, current_curv, x_ref, car_params, controller_config, controller_state)
            logger.warning('MPC solver failed for %s: %s',
                           car_params.name, info['return_status'])
            return (fallback_ctrl, True, controller_state, info['return_status'])

        steering = _clip(float(u_opt[0, 0]),
                         -car_params.max_steer_right, car_params.max_steer_left)
        throttle = _clip(float(u_opt[1, 0]),
                         car_params.min_throttle, car_params.max_throttle)
        ctrl = Control(steering=steering, throttle=throttle)

        controller_state.last_x = x_opt
        controller_state.last_u = u_opt
        controller_state.last_applied_u = np.array([steering, throttle], dtype=float)
        controller_state.reference_curv_traj = x_ref
        controller_state.predicted_curv_traj = x_opt
        controller_state.predicted_cart_traj = MpcController._curv_traj_to_cart(track, x_opt)
        controller_state.debug_dict = {
            'solver_status': info['return_status'],
            'solver_iterations': info['iterations'],
            'solver_objective': info['objective'],
            'current_state_curv': current_curv.tolist(),
            'first_ref_state': x_ref[:, 0].tolist(),
            'first_control': [steering, throttle],
        }

        main_state.car_target_v[car_index] = float(x_ref[3, min(1, x_ref.shape[1] - 1)])
        return (ctrl, True, controller_state, 'Controller OK')
