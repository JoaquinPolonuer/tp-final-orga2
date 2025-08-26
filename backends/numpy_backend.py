import numpy as np
from backends.backend import Backend


class NumpyBackend(Backend):
    def fft2(self, x):
        return np.fft.fft2(x)

    def ifft2(self, x):
        return np.fft.ifft2(x)

    def fftfreq(self, n, d):
        return np.fft.fftfreq(n, d)
