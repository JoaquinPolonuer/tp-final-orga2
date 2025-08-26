import numpy as np
from abc import ABC, abstractmethod


class Backend:
    def zeros(self, shape, dtype=complex):
        return np.zeros(shape, dtype=dtype)

    def linspace(self, start, stop, num):
        return np.linspace(start, stop, num)

    def meshgrid(self, x, y):
        return np.meshgrid(x, y)

    @abstractmethod
    def fft2(self, x):
        pass

    @abstractmethod
    def ifft2(self, x):
        pass

    @abstractmethod
    def fftfreq(self, n, d):
        pass

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
