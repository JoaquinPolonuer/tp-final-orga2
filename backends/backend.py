from abc import ABC, abstractmethod


class Backend(ABC):
    """Abstract backend interface"""

    @abstractmethod
    def zeros(self, shape, dtype=complex):
        pass

    @abstractmethod
    def linspace(self, start, stop, num):
        pass

    @abstractmethod
    def meshgrid(self, x, y):
        pass

    @abstractmethod
    def fft2(self, x):
        pass

    @abstractmethod
    def ifft2(self, x):
        pass

    @abstractmethod
    def fftfreq(self, n, d):
        pass

    @abstractmethod
    def abs(self, x):
        pass

    @abstractmethod
    def real(self, x):
        pass

    @abstractmethod
    def max(self, x):
        pass

    @abstractmethod
    def sum(self, x):
        pass

    @abstractmethod
    def clip(self, x, min_val, max_val):
        pass
