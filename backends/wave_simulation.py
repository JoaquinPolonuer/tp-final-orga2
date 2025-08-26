import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from abc import ABC, abstractmethod
import math
import cmath
import time
from backends.numpy_backend import NumpyBackend
from backends.pure_python_backend import PurePythonBackend
from backends.c_backend import CBackend
from utils import to_array

backends = {
    "numpy": NumpyBackend,
    "python": PurePythonBackend,
    "c": CBackend,
}


class WaveSimulation2D:
    def __init__(self, backend="numpy", size=256, domain_size=10.0, wave_speed=1.0, dt=0.01):

        self.backend = backends.get(backend, NumpyBackend)()

        self.size = size
        self.domain_size = domain_size
        self.wave_speed = wave_speed
        self.dx = domain_size / size
        self.dt = dt

        # Spatial grid
        self.X, self.Y = self.backend.meshgrid(
            self.backend.linspace(-domain_size / 2, domain_size / 2, size),
            self.backend.linspace(-domain_size / 2, domain_size / 2, size),
        )

        # Frequency grid (k-space)
        self.KX, self.KY = self.backend.meshgrid(
            self.backend.fftfreq(size, self.dx), self.backend.fftfreq(size, self.dx)
        )

        self.K = self._initialize_K(size=size)

        # Initialize wave field
        self.wave = self.backend.zeros((size, size), dtype=complex)
        self.wave_k = self.backend.fft2(self.wave)

    def add_wave_source(self, x_pos, y_pos, amplitude=1.0, frequency=3.0, width=0.5):
        """Add a wave source at specified position"""
        x_idx = int((x_pos + self.domain_size / 2) / self.domain_size * self.size)
        y_idx = int((y_pos + self.domain_size / 2) / self.domain_size * self.size)

        x_idx = self.backend.clip(x_idx, 0, self.size - 1)
        y_idx = self.backend.clip(y_idx, 0, self.size - 1)

        # Create wave pattern - simplified for backend compatibility
        if isinstance(self.backend, NumpyBackend):
            # Use full NumPy implementation
            envelope = amplitude * np.exp(
                -((np.array(self.X) - x_pos) ** 2 + (np.array(self.Y) - y_pos) ** 2) / width**2
            )
            r = np.sqrt((np.array(self.X) - x_pos) ** 2 + (np.array(self.Y) - y_pos) ** 2)
            phase = frequency * r
            new_wave = envelope * np.exp(1j * phase)
            self.wave += new_wave
        else:
            # Simplified for pure Python
            for i in range(self.size):
                for j in range(self.size):
                    x_val = self.X[i][j]
                    y_val = self.Y[i][j]
                    r_sq = (x_val - x_pos) ** 2 + (y_val - y_pos) ** 2
                    envelope = amplitude * math.exp(-r_sq / width**2)
                    r = math.sqrt(r_sq)
                    phase = frequency * r
                    new_val = envelope * cmath.exp(1j * phase)
                    self.wave[i][j] += new_val

        self.wave_k = self.backend.fft2(self.wave)

    def _initialize_K(self, size):
        # Calculate K magnitude
        if isinstance(self.backend, NumpyBackend):
            K = np.sqrt(np.array(self.KX) ** 2 + np.array(self.KY) ** 2) * 2 * np.pi
            K[0, 0] = 1e-10
            return K
        else:
            # For C backend and pure Python backend
            K = []
            for i in range(size):
                row = []
                for j in range(size):
                    kx_val = self.KX[i][j]
                    ky_val = self.KY[i][j]
                    k_mag = math.sqrt(kx_val**2 + ky_val**2) * 2 * math.pi
                    if i == 0 and j == 0:
                        k_mag = 1e-10
                    row.append(k_mag)
                K.append(row)
            return K

    def step(self):
        """Advance simulation by one time step"""
        if isinstance(self.backend, NumpyBackend):
            omega = self.wave_speed * self.K
            phase_factor = np.exp(-1j * omega * self.dt)
            self.wave_k *= phase_factor
        else:
            # Simplified time evolution for pure Python
            for i in range(self.size):
                for j in range(self.size):
                    omega = self.wave_speed * self.K[i][j]
                    phase_factor = cmath.exp(-1j * omega * self.dt)
                    self.wave_k[i][j] *= phase_factor

        self.wave = self.backend.ifft2(self.wave_k)

    def get_intensity(self):
        """Get wave intensity for visualization"""
        if isinstance(self.backend, NumpyBackend):
            return self.backend.abs(self.wave) ** 2
        else:
            # For C backend and pure Python backend
            return [[abs(self.wave[i][j]) ** 2 for j in range(self.size)] for i in range(self.size)]

    def get_real_part(self):
        """Get real part of wave for visualization"""
        return self.backend.real(self.wave)
