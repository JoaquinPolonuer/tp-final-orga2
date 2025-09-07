import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from backends.wave_simulation_numpy import NumpyWaveSimulation2D
from backends.wave_simulation_python import PythonWaveSimulation2D
from backends.wave_simulation_c import CWaveSimulation2D
from backends.wave_simulation_c_avx import OptimizedCWaveSimulation2D
from backends.wave_simulation_asm import ASMWaveSimulation2D
from backends.wave_simulation_asm_simd import ASMSIMDWaveSimulation2D


class WaveVisualizer:
    def __init__(self, simulation):
        self.sim = simulation
        self.backend_name = self.sim.__class__.__name__.replace("WaveSimulation2D", "").lower()
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
            vmax=0.5,
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
            self.ax1.set_title(f"Wave Intensity |ψ|² ({self.backend_name} backend) - FPS: {self.fps_counter['last_fps']:.1f}")

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
    # sim = ASMSIMDWaveSimulation2D(
    #     size=128,
    #     domain_size=8.0,
    #     wave_speed=2.0,
    #     dt=0.02,
    # )

    sim = ASMWaveSimulation2D(
        size=128,
        domain_size=8.0,
        wave_speed=2.0,
        dt=0.02,
    )

    visualizer = WaveVisualizer(sim)
    ani = visualizer.run()
