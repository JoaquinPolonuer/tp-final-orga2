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
        self.dt = dt

        # Spatial grid
        self.dx = domain_size / size
        self.X, self.Y = self.backend.meshgrid(
            self.backend.linspace(-domain_size / 2, domain_size / 2, size),
            self.backend.linspace(-domain_size / 2, domain_size / 2, size),
        )

        # Frequency grid (k-space)
        self.KX, self.KY = self.backend.meshgrid(
            self.backend.fftfreq(size, self.dx), self.backend.fftfreq(size, self.dx)
        )

        # Calculate K magnitude
        if isinstance(self.backend, NumpyBackend):
            self.K = self.backend.sqrt(np.array(self.KX) ** 2 + np.array(self.KY) ** 2) * 2 * np.pi
            self.K[0, 0] = 1e-10
        elif isinstance(self.backend, CBackend):
            # C backend: calculate K magnitude without NumPy
            self.K = []
            for i in range(size):
                row = []
                for j in range(size):
                    kx_val = self.KX[i][j]
                    ky_val = self.KY[i][j]
                    k_mag = math.sqrt(kx_val**2 + ky_val**2) * 2 * math.pi
                    if i == 0 and j == 0:
                        k_mag = 1e-10
                    row.append(k_mag)
                self.K.append(row)
        else:
            # For pure Python, simplified K calculation
            self.K = [
                [
                    (
                        1e-10
                        if i == 0 and j == 0
                        else math.sqrt(self.KX[i][j] ** 2 + self.KY[i][j] ** 2) * 2 * math.pi
                    )
                    for j in range(size)
                ]
                for i in range(size)
            ]

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


def run_simulation(backend="numpy", simulation_size=128):
    """Run simulation with specified backend"""
    print(f"Running simulation with {backend} backend")

    # Create simulation
    sim = WaveSimulation2D(
        backend=backend,
        size=simulation_size,
        domain_size=8.0,
        wave_speed=2.0,
        dt=0.02,
    )

    # Set up visualization with FPS counter
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # FPS tracking
    fps_counter = {"frame_count": 0, "start_time": time.time(), "last_fps": 0.0}

    im1 = ax1.imshow(
        to_array(sim.get_intensity()),
        extent=[-4, 4, -4, 4],
        cmap="hot",
        vmin=0,
        vmax=2,
        origin="lower",
    )
    ax1.set_title(f"Wave Intensity |ψ|² ({backend} backend) - FPS: 0.0")
    plt.colorbar(im1, ax=ax1)

    im2 = ax2.imshow(
        to_array(sim.get_real_part()),
        extent=[-4, 4, -4, 4],
        cmap="RdBu",
        vmin=-1,
        vmax=1,
        origin="lower",
    )
    ax2.set_title(f"Wave Real Part Re(ψ) ({backend} backend)")
    plt.colorbar(im2, ax=ax2)

    def on_click(event):
        if event.inaxes in [ax1, ax2] and event.button == 1:
            x_click, y_click = event.xdata, event.ydata
            if x_click is not None and y_click is not None:
                sim.add_wave_source(x_click, y_click, amplitude=0.8, frequency=4.0)

    fig.canvas.mpl_connect("button_press_event", on_click)

    def animate(frame):
        sim.step()

        # Update FPS counter
        fps_counter["frame_count"] += 1
        current_time = time.time()
        elapsed = current_time - fps_counter["start_time"]

        if elapsed >= 1.0:  # Update FPS every second
            fps_counter["last_fps"] = fps_counter["frame_count"] / elapsed
            fps_counter["frame_count"] = 0
            fps_counter["start_time"] = current_time

            # Update title with FPS
            ax1.set_title(
                f"Wave Intensity |ψ|² ({backend} backend) - FPS: {fps_counter['last_fps']:.1f}"
            )

        intensity = to_array(sim.get_intensity())
        real_part = to_array(sim.get_real_part())

        max_intensity = np.max(intensity)
        if max_intensity > 0:
            im1.set_clim(0, max_intensity * 1.1)

        max_real = np.max(np.abs(real_part))
        if max_real > 0:
            im2.set_clim(-max_real * 1.1, max_real * 1.1)

        im1.set_data(intensity)
        im2.set_data(real_part)

        return [im1, im2]

    ani = animation.FuncAnimation(fig, animate, frames=1000, interval=50, blit=False)
    plt.tight_layout()
    plt.show()

    return sim, ani


if __name__ == "__main__":
    # Choose backend: 'numpy', 'python', or 'c'
    backend_choice = "numpy"  # Change to 'python' to test pure Python backend, 'c' for C backend

    simulation, anim = run_simulation(backend=backend_choice, simulation_size=128)

    print(f"Backend: {backend_choice}")
    print(f"Grid size: {simulation.size}x{simulation.size}")
    print(f"Domain size: {simulation.domain_size}")
