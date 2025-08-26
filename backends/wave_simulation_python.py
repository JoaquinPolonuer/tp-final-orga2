import numpy as np
import math
import cmath

class PythonWaveSimulation2D:
    def __init__(self, size=256, domain_size=10.0, wave_speed=1.0, dt=0.01):

        self.size = size
        self.domain_size = domain_size
        self.wave_speed = wave_speed
        self.dx = domain_size / size
        self.dt = dt

        # Spatial grid - store (X,Y) tuples at each grid point
        x_coords = np.linspace(-domain_size / 2, domain_size / 2, size)
        y_coords = np.linspace(-domain_size / 2, domain_size / 2, size)
        self.grid = [[(x_coords[j], y_coords[i]) for j in range(size)] for i in range(size)]

        # Frequency grid (k-space) - store (KX,KY) tuples at each grid point
        k_coords = self.fftfreq(size, self.dx)
        self.k_grid = [[(k_coords[j], k_coords[i]) for j in range(size)] for i in range(size)]

        self.K = self._initialize_K(size=size)

        # Initialize wave field
        self.wave = np.zeros((size, size), dtype=complex)
        self.wave_k = self.fft2(self.wave)

    def add_wave_source(self, x_pos, y_pos, amplitude=1.0, frequency=3.0, width=0.5):
        """Add a wave source at specified position"""
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

    def _initialize_K(self, size):
        K = []
        for i in range(size):
            row = []
            for j in range(size):
                kx_val, ky_val = self.k_grid[i][j]
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

    def _fft_1d(self, x: list[complex]) -> list[complex]:
        """1D FFT using Cooley-Tukey algorithm"""
        n = len(x)
        if n <= 1:
            return x[:]

        # Pad to power of 2 if needed
        if n & (n - 1) != 0:
            next_power = 1 << (n - 1).bit_length()
            x = x + [0 + 0j] * (next_power - n)
            n = next_power

        # Bit-reversal permutation
        j = 0
        for i in range(1, n):
            bit = n >> 1
            while j & bit:
                j ^= bit
                bit >>= 1
            j ^= bit
            if i < j:
                x[i], x[j] = x[j], x[i]

        # Cooley-Tukey FFT
        length = 2
        while length <= n:
            w = cmath.exp(-2j * math.pi / length)
            for i in range(0, n, length):
                wn = 1 + 0j
                for j in range(length // 2):
                    u = x[i + j]
                    v = x[i + j + length // 2] * wn
                    x[i + j] = u + v
                    x[i + j + length // 2] = u - v
                    wn *= w
            length <<= 1

        return x

    def _ifft_1d(self, x: list[complex]) -> list[complex]:
        """1D IFFT using FFT"""
        n = len(x)
        # Conjugate input
        x_conj = [val.conjugate() for val in x]
        # Apply FFT
        result = self._fft_1d(x_conj)
        # Conjugate output and normalize
        return [val.conjugate() / n for val in result]

    def fft2(self, x):
        """2D FFT using row-column method"""
        rows = len(x)
        cols = len(x[0]) if rows > 0 else 0

        # FFT on rows
        row_fft = []
        for i in range(rows):
            row_fft.append(self._fft_1d(x[i][:]))

        # FFT on columns
        result = []
        for i in range(rows):
            result.append([0 + 0j] * cols)

        for j in range(cols):
            col = [row_fft[i][j] for i in range(rows)]
            col_fft = self._fft_1d(col)
            for i in range(rows):
                result[i][j] = col_fft[i]

        return result

    def ifft2(self, x):
        """2D IFFT using row-column method"""
        rows = len(x)
        cols = len(x[0]) if rows > 0 else 0

        # IFFT on rows
        row_ifft = []
        for i in range(rows):
            row_ifft.append(self._ifft_1d(x[i][:]))

        # IFFT on columns
        result = []
        for i in range(rows):
            result.append([0 + 0j] * cols)

        for j in range(cols):
            col = [row_ifft[i][j] for i in range(rows)]
            col_ifft = self._ifft_1d(col)
            for i in range(rows):
                result[i][j] = col_ifft[i]

        return result

    def fftfreq(self, n: int, d: float = 1.0) -> list[float]:
        """Generate frequency array for FFT"""
        result = []
        for i in range(n):
            if i <= n // 2:
                result.append(i / (n * d))
            else:
                result.append((i - n) / (n * d))
        return result
