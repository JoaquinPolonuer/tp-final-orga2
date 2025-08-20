import math
import cmath
from backends.backend import Backend

try:
    from backends import c_backend_core
except ImportError:
    raise ImportError(
        "C backend core module not available. "
        "Please compile it using: python setup.py build_ext --inplace"
    )


class CBackend(Backend):
    """C backend implementation using optimized C functions for critical operations"""

    def __init__(self):
        self.c_core = c_backend_core

    def zeros(self, shape, dtype=complex):
        if isinstance(shape, int):
            shape = (shape,)
        if len(shape) == 2:
            return self.c_core.zeros(shape[0], shape[1])
        else:
            # Fallback to pure Python for other shapes
            if len(shape) == 1:
                return [0 + 0j if dtype == complex else 0] * shape[0]
            return None  # Not implemented for higher dimensions

    def linspace(self, start, stop, num):
        return self.c_core.linspace(start, stop, num)

    def meshgrid(self, x, y):
        return self.c_core.meshgrid(x, y)

    def sqrt(self, x):
        # Pure Python implementation
        if isinstance(x, (list, tuple)):
            result = []
            for i in range(len(x)):
                if isinstance(x[i], (list, tuple)):
                    row = []
                    for j in range(len(x[i])):
                        val = x[i][j]
                        if isinstance(val, complex):
                            row.append(math.sqrt(val.real**2 + val.imag**2))
                        else:
                            row.append(math.sqrt(abs(val)))
                    result.append(row)
                else:
                    val = x[i]
                    if isinstance(val, complex):
                        result.append(math.sqrt(val.real**2 + val.imag**2))
                    else:
                        result.append(math.sqrt(abs(val)))
            return result
        if isinstance(x, complex):
            return math.sqrt(x.real**2 + x.imag**2)
        return math.sqrt(abs(x))

    def exp(self, x):
        # Pure Python implementation
        if isinstance(x, (list, tuple)):
            result = []
            for i in range(len(x)):
                if isinstance(x[i], (list, tuple)):
                    row = []
                    for j in range(len(x[i])):
                        val = x[i][j]
                        if isinstance(val, complex):
                            row.append(cmath.exp(val))
                        else:
                            row.append(math.exp(val))
                    result.append(row)
                else:
                    val = x[i]
                    if isinstance(val, complex):
                        result.append(cmath.exp(val))
                    else:
                        result.append(math.exp(val))
            return result
        if isinstance(x, complex):
            return cmath.exp(x)
        return math.exp(x)

    def fft2(self, x):
        return self.c_core.fft2(x)

    def ifft2(self, x):
        return self.c_core.ifft2(x)

    def fftfreq(self, n, d):
        return self.c_core.fftfreq(n, d)

    def abs(self, x):
        return self.c_core.abs_array(x)

    def real(self, x):
        return self.c_core.real_array(x)

    def max(self, x):
        # Pure Python implementation
        if isinstance(x, (list, tuple)):
            if isinstance(x[0], (list, tuple)):
                # 2D array
                max_val = float("-inf")
                for row in x:
                    for val in row:
                        if val > max_val:
                            max_val = val
                return max_val
            else:
                # 1D array
                return max(x)
        return x

    def sum(self, x):
        # Pure Python implementation
        if isinstance(x, (list, tuple)):
            if isinstance(x[0], (list, tuple)):
                # 2D array
                total = 0
                for row in x:
                    for val in row:
                        total += val
                return total
            else:
                # 1D array
                return sum(x)
        return x

    def clip(self, x, min_val, max_val):
        # Pure Python implementation
        return max(min_val, min(max_val, x))
