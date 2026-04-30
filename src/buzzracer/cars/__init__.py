"""Lazy exports for car implementations.

Avoid importing runtime-heavy car modules when callers only need lightweight
submodules such as ``buzzracer.cars.car_param``.
"""

from importlib import import_module

__all__ = ['OldOffboard', 'Offboard', 'FHSS']

_LAZY_IMPORTS = {
    'OldOffboard': '.old_offboard',
    'Offboard': '.offboard',
    'FHSS': '.fhss',
}


def __getattr__(name):
    module_name = _LAZY_IMPORTS.get(name)
    if module_name is None:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')

    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
