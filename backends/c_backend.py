import numpy as np
import math
import cmath
from backends.backend import Backend

try:
    from backends import pure_c_backend_core
except ImportError:
    raise ImportError(
        "Pure C backend core module not available. "
        "Please compile it using: python setup.py build_ext --inplace"
    )


class CBackend(Backend):
    def __init__(self):
        self.c_core = pure_c_backend_core

    def fft2(self, x):
        return self.c_core.fft2([list(row) for row in x])

    def ifft2(self, x):
        return self.c_core.ifft2([list(row) for row in x])

    def fftfreq(self, n, d):
        return self.c_core.fftfreq(n, d)
