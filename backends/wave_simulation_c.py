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

        self._initialize_grid()

        self._initialize_K_grid()

        self._initialize_K()

        self._initialize_wave()
        
        self._initialize_wave_k()

    def _initialize_wave(self):
        self.wave = np.zeros((self.size, self.size), dtype=complex)

    def _initialize_wave_k(self):
        self.wave_k = self.fft2(self.wave)

    def _initialize_grid(self):
        x_coords = np.linspace(-self.domain_size / 2, self.domain_size / 2, self.size)
        y_coords = np.linspace(-self.domain_size / 2, self.domain_size / 2, self.size)
        self.grid = [
            [(x_coords[j], y_coords[i]) for j in range(self.size)] for i in range(self.size)
        ]

    def _initialize_K_grid(self):
        # Frequency grid (k-space) - store (KX,KY) tuples at each grid point
        k_coords = self.fftfreq(self.size, self.dx)
        self.k_grid = [
            [(k_coords[j], k_coords[i]) for j in range(self.size)] for i in range(self.size)
        ]

    def add_wave_source(self, x_pos, y_pos, amplitude=1.0, frequency=3.0, width=0.5):
        x_idx = int((x_pos + self.domain_size / 2) / self.domain_size * self.size)
        y_idx = int((y_pos + self.domain_size / 2) / self.domain_size * self.size)

        x_idx = np.clip(x_idx, 0, self.size - 1)
        y_idx = np.clip(y_idx, 0, self.size - 1)

        for i in range(self.size):
            for j in range(self.size):
                x_val, y_val = self.grid[i][j]
                r_sq = (x_val - x_pos) ** 2 + (y_val - y_pos) ** 2
                envelope = amplitude * math.exp(-r_sq / width**2)
                r = math.sqrt(r_sq)
                phase = frequency * r
                new_val = envelope * cmath.exp(1j * phase)
                self.wave[i][j] += new_val

        self.wave_k = self.fft2(self.wave)

    def _initialize_K(self):
        self.K = []
        for i in range(self.size):
            row = []
            for j in range(self.size):
                kx_val, ky_val = self.k_grid[i][j]
                k_mag = math.sqrt(kx_val**2 + ky_val**2) * 2 * math.pi
                if i == 0 and j == 0:
                    k_mag = 1e-10
                row.append(k_mag)
            self.K.append(row)

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
