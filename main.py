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
from backends.wave_simulation import WaveSimulation2D

backends = {
    "numpy": NumpyBackend,
    "python": PurePythonBackend,
    "c": CBackend,
}


class WaveVisualizer:
    def __init__(self, simulation, backend_name):
        self.sim = simulation
        self.backend_name = backend_name
        self.fps_counter = {"frame_count": 0, "start_time": time.time(), "last_fps": 0.0}

        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 5))
        self._setup_plots()
        self._connect_events()

    def _setup_plots(self):
        self.im1 = self.ax1.imshow(
            self.sim.get_intensity(),
            extent=[-4, 4, -4, 4],
            cmap="hot",
            vmin=0,
            vmax=2,
            origin="lower",
        )
        self.ax1.set_title(f"Wave Intensity |ψ|² ({self.backend_name} backend) - FPS: 0.0")
        plt.colorbar(self.im1, ax=self.ax1)

        self.im2 = self.ax2.imshow(
            self.sim.get_real_part(),
            extent=[-4, 4, -4, 4],
            cmap="RdBu",
            vmin=-0.5,
            vmax=0.5,
            origin="lower",
        )
        self.ax2.set_title(f"Wave Real Part Re(ψ) ({self.backend_name} backend)")
        plt.colorbar(self.im2, ax=self.ax2)

    def _connect_events(self):
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)

    def _on_click(self, event):
        if event.inaxes in [self.ax1, self.ax2] and event.button == 1:
            x_click, y_click = event.xdata, event.ydata
            if x_click is not None and y_click is not None:
                self.sim.add_wave_source(x_click, y_click, amplitude=0.8, frequency=4.0)

    def _update_fps(self):
        self.fps_counter["frame_count"] += 1
        current_time = time.time()
        elapsed = current_time - self.fps_counter["start_time"]

        if elapsed >= 1.0:
            self.fps_counter["last_fps"] = self.fps_counter["frame_count"] / elapsed
            self.fps_counter["frame_count"] = 0
            self.fps_counter["start_time"] = current_time
            self.ax1.set_title(
                f"Wave Intensity |ψ|² ({self.backend_name} backend) - FPS: {self.fps_counter['last_fps']:.1f}"
            )

    def animate(self, frame):
        self.sim.step()

        self._update_fps()

        intensity = self.sim.get_intensity()
        real_part = self.sim.get_real_part()

        self.im1.set_data(intensity)
        self.im2.set_data(real_part)

        return [self.im1, self.im2]

    def run(self):
        ani = animation.FuncAnimation(self.fig, self.animate, frames=1000, interval=50, blit=False)
        plt.tight_layout()
        plt.show()
        return ani


if __name__ == "__main__":
    # Choose backend: 'numpy', 'python', or 'c'
    backend_name = "c"  # Change to 'python' to test pure Python backend, 'c' for C backend

    sim = WaveSimulation2D(
        backend=backend_name,
        size=128,
        domain_size=8.0,
        wave_speed=2.0,
        dt=0.02,
    )

    visualizer = WaveVisualizer(sim, backend_name)
    ani = visualizer.run()
