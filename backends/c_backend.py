import numpy as np
from backends.backend import Backend

try:
    from backends import c_backend_core
except ImportError:
    raise ImportError(
        "C backend core module not available. "
        "Please compile it using: python setup.py build_ext --inplace"
    )


class CBackend(Backend):
    """C backend implementation using optimized C functions"""

    def __init__(self):
        self.c_core = c_backend_core

    def zeros(self, shape, dtype=complex):
        if isinstance(shape, int):
            shape = (shape,)
        if len(shape) == 1:
            return np.zeros(shape, dtype=dtype)
        elif len(shape) == 2:
            return self.c_core.zeros(shape[0], shape[1])
        else:
            return np.zeros(shape, dtype=dtype)

    def linspace(self, start, stop, num):
        return self.c_core.linspace(start, stop, num)

    def meshgrid(self, x, y):
        return self.c_core.meshgrid(x, y)

    def sqrt(self, x):
        return np.sqrt(x)

    def exp(self, x):
        return np.exp(x)

    def fft2(self, x):
        return self.c_core.fft2(x)

    def ifft2(self, x):
        return self.c_core.ifft2(x)

    def fftfreq(self, n, d):
        return self.c_core.fftfreq(n, d)

    def abs(self, x):
        if isinstance(x, np.ndarray) and np.iscomplexobj(x):
            return self.c_core.abs_array(x)
        return np.abs(x)

    def real(self, x):
        if isinstance(x, np.ndarray) and np.iscomplexobj(x):
            return self.c_core.real_array(x)
        return np.real(x)

    def max(self, x):
        return np.max(x)

    def sum(self, x):
        return np.sum(x)

    def clip(self, x, min_val, max_val):
        return np.clip(x, min_val, max_val)
