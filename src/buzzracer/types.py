
''' Define types used througout the project'''
import ctypes
from math import isnan


class Control(ctypes.Structure):
    steering: float
    ''' Steering angle for an Ackermann steering vehicle in rad, left positive '''
    throttle: float
    ''' Throttle, positive forward, negative braking. Typical rance [-1,1]
     In some VehicleDynamics, this is mapped to acceleration directly, 
     but in general does not have direct physical meaning '''

    _fields_ = [
        ('steering', ctypes.c_double),
        ('throttle', ctypes.c_double)
    ]

    def __init__(self, *args, **kwargs):
        # 1. Let ctypes handle standard field initialization first
        super().__init__(*args, **kwargs)

        # 2. Run the validation check (acting as a post-init)
        self._check_for_nans()

    def _check_for_nans(self):
        """Iterates through defined fields and raises ValueError if any are NaN."""
        for field_name, _ in self._fields_:
            # getattr retrieves the Python float value from the c_double field
            val = getattr(self, field_name)
            if isnan(val):
                raise ValueError(f"Invalid initialization: Attribute '{field_name}' cannot be NaN.")

    def to_tuple(self) -> tuple:
        """Returns the control as a standard Python tuple."""
        return (self.steering, self.throttle)

    def __repr__(self):
        return f"Control(steering={self.steering:.3f}, throttle={self.throttle:.3f})"


class StateTiming(ctypes.Structure):
    """Timing metadata associated with a published state sample."""

    _fields_ = [
        ('seq', ctypes.c_ulonglong),
        ('udp_rx_ts', ctypes.c_double),
        ('rigid_body_ts', ctypes.c_double),
        ('car_state_ts', ctypes.c_double)
    ]


class ControlTiming(ctypes.Structure):
    """Timing metadata associated with a published control sample."""

    _fields_ = [
        ('seq', ctypes.c_ulonglong),
        ('udp_rx_ts', ctypes.c_double),
        ('rigid_body_ts', ctypes.c_double),
        ('car_state_ts', ctypes.c_double),
        ('controller_read_ts', ctypes.c_double),
        ('controller_done_ts', ctypes.c_double)
    ]


class CartesianState(ctypes.Structure):
    x: float
    ''' X coordinate in global frame, unit: m'''
    y: float
    ''' Y coordinate in global frame, unit: m'''
    heading: float
    ''' Heading in radians from x axis, ccw positive'''
    v_forward: float
    ''' Longitudinal speed m/s w.r.t. vehicle centerline, forward positive '''
    v_sideway: float
    ''' Lateral speed m/s w.r.t. vehicle centerline, left positive '''
    omega: float

    _fields_ = [
        ('x', ctypes.c_double),
        ('y', ctypes.c_double),
        ('heading', ctypes.c_double),
        ('v_forward', ctypes.c_double),
        ('v_sideway', ctypes.c_double),
        ('omega', ctypes.c_double)
    ]

    def __init__(self, *args, **kwargs):
        # 1. Let ctypes handle standard field initialization first
        super().__init__(*args, **kwargs)

        # 2. Run the validation check (acting as a post-init)
        self._check_for_nans()

    def _check_for_nans(self):
        """Iterates through defined fields and raises ValueError if any are NaN."""
        for field_name, _ in self._fields_:
            # getattr retrieves the Python float value from the c_double field
            val = getattr(self, field_name)
            if isnan(val):
                raise ValueError(f"Invalid initialization: Attribute '{field_name}' cannot be NaN.")

    def to_tuple(self) -> tuple:
        """Returns the state as a standard Python tuple."""
        return (self.x, self.y, self.heading, self.v_forward, self.v_sideway, self.omega)

    def __repr__(self):
        """Makes printing the struct look just like the NamedTuple."""
        return (f"CartesianState(x={self.x:.3f}, y={self.y:.3f}, "
                f"heading={self.heading:.3f}, v_forward={self.v_forward:.3f}, "
                f"v_sideway={self.v_sideway:.3f}, omega={self.omega:.3f})")

    def __getitem__(self, index):
        """Makes the struct subscriptable (e.g., state[0]) and unpackable."""
        if isinstance(index, int):
            # Handle standard integer indexing
            if index < 0 or index >= len(self._fields_):
                raise IndexError("CartesianState index out of range")
            field_name = self._fields_[index][0]
            return getattr(self, field_name)

        elif isinstance(index, slice):
            # Handle slicing (e.g., state[0:3] to get x, y, heading)
            return tuple(getattr(self, self._fields_[i][0]) for i in range(*index.indices(len(self._fields_))))

        else:
            raise TypeError("CartesianState indices must be integers or slices")


class CurvilinearState(ctypes.Structure):
    ''' A state in Frenet frame'''
    progress: float
    ''' Progress along reference curve'''
    lateral_err: float
    ''' Lateral deviation, left positive'''
    heading_err: float
    ''' Relative heading w.r.t. reference curve'''
    v_forward: float
    ''' Longitudinal speed, forward positive '''
    v_sideway: float
    ''' Lateral speed, left positive '''
    rel_omega: float = 0
    ''' Relative angular velocity w.r.t. desired on ref curve at current speed, ccw positive'''

    _fields_ = [
        ('progress', ctypes.c_double),
        ('lateral_err', ctypes.c_double),
        ('heading_err', ctypes.c_double),
        ('v_forward', ctypes.c_double),
        ('v_sideway', ctypes.c_double),
        ('rel_omega', ctypes.c_double)
    ]

    def __init__(self, *args, **kwargs):
        # 1. Let ctypes handle standard field initialization first
        super().__init__(*args, **kwargs)

        # 2. Run the validation check (acting as a post-init)
        self._check_for_nans()

    def _check_for_nans(self):
        """Iterates through defined fields and raises ValueError if any are NaN."""
        for field_name, _ in self._fields_:
            # getattr retrieves the Python float value from the c_double field
            val = getattr(self, field_name)
            if isnan(val):
                raise ValueError(f"Invalid initialization: Attribute '{field_name}' cannot be NaN.")

    def to_tuple(self) -> tuple:
        """Returns the curvilinear state as a standard Python tuple."""
        return (self.progress, self.lateral_err, self.heading_err,
                self.v_forward, self.v_sideway, self.rel_omega)

    def __repr__(self):
        return (f"CurvilinearState(progress={self.progress:.3f}, lateral_err={self.lateral_err:.3f}, "
                f"heading_err={self.heading_err:.3f}, v_forward={self.v_forward:.3f}, "
                f"v_sideway={self.v_sideway:.3f}, rel_omega={self.rel_omega:.3f})")
