import numpy as np
import math
import cmath
try:
    from backends import c_backend_core
except ImportError:
    raise ImportError(
        "Pure C backend core module not available. "
        "Please compile it using: python setup.py build_ext --inplace"
    )


class CWaveSimulation2D:
    def __init__(self, size=256, domain_size=10.0, wave_speed=1.0, dt=0.01):

        self.c_core = c_backend_core

        self.size = size
        self.domain_size = domain_size
        self.wave_speed = wave_speed
        self.dx = domain_size / size
        self.dt = dt

        # Spatial grid
        self.X, self.Y = np.meshgrid(
            np.linspace(-domain_size / 2, domain_size / 2, size),
            np.linspace(-domain_size / 2, domain_size / 2, size),
        )

        # Frequency grid (k-space)
        self.KX, self.KY = np.meshgrid(self.fftfreq(size, self.dx), self.fftfreq(size, self.dx))

        self.K = self._initialize_K(size=size)

        # Initialize wave field
        self.wave = np.zeros((size, size), dtype=complex)
        self.wave_k = self.fft2(self.wave)

    def add_wave_source(self, x_pos, y_pos, amplitude=1.0, frequency=3.0, width=0.5):
        x_idx = int((x_pos + self.domain_size / 2) / self.domain_size * self.size)
        y_idx = int((y_pos + self.domain_size / 2) / self.domain_size * self.size)

        x_idx = np.clip(x_idx, 0, self.size - 1)
        y_idx = np.clip(y_idx, 0, self.size - 1)

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

        self.wave_k = self.fft2(self.wave)

    def _initialize_K(self, size):
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
        for i in range(self.size):
            for j in range(self.size):
                omega = self.wave_speed * self.K[i][j]
                phase_factor = cmath.exp(-1j * omega * self.dt)
                self.wave_k[i][j] *= phase_factor

        self.wave = self.ifft2(self.wave_k)

    def get_intensity(self):
        return [[abs(self.wave[i][j]) ** 2 for j in range(self.size)] for i in range(self.size)]

    def get_real_part(self):
        return np.real(self.wave)

    def fft2(self, x):
        return self.c_core.fft2([list(row) for row in x])

    def ifft2(self, x):
        return self.c_core.ifft2([list(row) for row in x])

    def fftfreq(self, n, d):
        return self.c_core.fftfreq(n, d)
