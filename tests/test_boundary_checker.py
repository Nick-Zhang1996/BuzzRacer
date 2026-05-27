from types import ModuleType, SimpleNamespace
import importlib.util
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import buzzracer
from buzzracer.types import CartesianState


def _load_module(module_name, relative_path):
    spec = importlib.util.spec_from_file_location(module_name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


if 'buzzracer.extensions' not in sys.modules:
    extensions_pkg = ModuleType('buzzracer.extensions')
    extensions_pkg.__path__ = [str(SRC / 'buzzracer' / 'extensions')]
    sys.modules['buzzracer.extensions'] = extensions_pkg
    buzzracer.extensions = extensions_pkg

extension_module = _load_module('buzzracer.extensions.extension', 'src/buzzracer/extensions/extension.py')
boundary_checker_module = _load_module(
    'buzzracer.extensions.boundary_checker',
    'src/buzzracer/extensions/boundary_checker.py',
)

BoundaryChecker = boundary_checker_module.BoundaryChecker
BoundaryCheckerConfig = boundary_checker_module.BoundaryCheckerConfig
BoundaryCheckerState = boundary_checker_module.BoundaryCheckerState
Extension = extension_module.Extension


class _FakeTrack:
    def __init__(self):
        self.data = SimpleNamespace(discretized_raceline=None)

    def precise_track_boundary(self, car_coord, car_heading):
        del car_heading
        return (-0.1, 0.2) if car_coord[0] > 1.0 else (0.2, 0.2)


class _FakeCar:
    def __init__(self, car_id, state):
        self.id = car_id
        self.state = state
        self.sim_state = CartesianState(*state)


def test_boundary_checker_resets_to_last_in_bounds_state():
    car = _FakeCar(
        0,
        CartesianState(x=0.0, y=1.0, heading=0.3, v_forward=0.4, v_sideway=0.1, omega=0.2),
    )
    main = SimpleNamespace(
        cars=[car],
        track=_FakeTrack(),
        state=SimpleNamespace(car_states=[CartesianState(*car.state)]),
    )
    Extension.main = main

    config = BoundaryCheckerConfig(SimpleNamespace())
    config.reset_to_last_in_boundary_state = True
    state = BoundaryCheckerState(config)
    checker = BoundaryChecker(config, state)

    car.state = CartesianState(x=2.0, y=3.0, heading=1.2, v_forward=1.5, v_sideway=0.6, omega=0.7)
    car.sim_state = CartesianState(*car.state)

    checker.update()

    assert car.state.x == 0.0
    assert car.state.y == 1.0
    assert car.state.heading == 0.3
    assert car.state.v_forward == 0.0
    assert car.state.v_sideway == 0.0
    assert car.state.omega == 0.0
    assert main.state.car_states[0].x == 0.0
    assert state.collision_count[car] == 1
