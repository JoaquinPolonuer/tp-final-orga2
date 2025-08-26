import numpy as np


class NumpyWaveSimulation2D:
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

        self.K = self._initialize_K()

        # Initialize wave field
        self.wave = np.zeros((size, size), dtype=complex)
        self.wave_k = self.fft2(self.wave)

    def fft2(self, x):
        return np.fft.fft2(x)

    def ifft2(self, x):
        return np.fft.ifft2(x)

    def fftfreq(self, n, d):
        return np.fft.fftfreq(n, d)

    def add_wave_source(self, x_pos, y_pos, amplitude=1.0, frequency=3.0, width=0.5):
        """Add a wave source at specified position"""
        x_idx = int((x_pos + self.domain_size / 2) / self.domain_size * self.size)
        y_idx = int((y_pos + self.domain_size / 2) / self.domain_size * self.size)

        x_idx = np.clip(x_idx, 0, self.size - 1)
        y_idx = np.clip(y_idx, 0, self.size - 1)

        # Create coordinate arrays from tuple grid for vectorized operations
        X = np.array([[self.grid[i][j][0] for j in range(self.size)] for i in range(self.size)])
        Y = np.array([[self.grid[i][j][1] for j in range(self.size)] for i in range(self.size)])
        
        envelope = amplitude * np.exp(-((X - x_pos) ** 2 + (Y - y_pos) ** 2) / width**2)
        r = np.sqrt((X - x_pos) ** 2 + (Y - y_pos) ** 2)
        phase = frequency * r
        new_wave = envelope * np.exp(1j * phase)
        self.wave += new_wave

        self.wave_k = self.fft2(self.wave)

    def _initialize_K(self):
        # Create coordinate arrays from tuple grid for vectorized operations
        KX = np.array([[self.k_grid[i][j][0] for j in range(self.size)] for i in range(self.size)])
        KY = np.array([[self.k_grid[i][j][1] for j in range(self.size)] for i in range(self.size)])
        
        K = np.sqrt(KX ** 2 + KY ** 2) * 2 * np.pi
        K[0, 0] = 1e-10
        return K

    def step(self):
        omega = self.wave_speed * self.K
        phase_factor = np.exp(-1j * omega * self.dt)
        self.wave_k *= phase_factor
        self.wave = self.ifft2(self.wave_k)

    def get_intensity(self):
        return np.abs(self.wave) ** 2

    def get_real_part(self):
        return np.real(self.wave)
