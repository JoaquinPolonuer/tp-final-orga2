import numpy as np
from backends.backend import Backend


class NumpyBackend(Backend):
    """NumPy backend implementation"""

    def zeros(self, shape, dtype=complex):
        return np.zeros(shape, dtype=dtype)

    def linspace(self, start, stop, num):
        return np.linspace(start, stop, num)

    def meshgrid(self, x, y):
        return np.meshgrid(x, y)

    def sqrt(self, x):
        return np.sqrt(x)

    def exp(self, x):
        return np.exp(x)

    def fft2(self, x):
        return np.fft.fft2(x)

    def ifft2(self, x):
        return np.fft.ifft2(x)

    def fftfreq(self, n, d):
        return np.fft.fftfreq(n, d)

    def abs(self, x):
        return np.abs(x)

    def real(self, x):
        return np.real(x)

    def max(self, x):
        return np.max(x)

    def sum(self, x):
        return np.sum(x)

    def clip(self, x, min_val, max_val):
        return np.clip(x, min_val, max_val)
