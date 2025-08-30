try:
    from backends import c_backend_asm
except ImportError:
    raise ImportError(
        "Pure C backend core module not available. "
        "Please compile it using: python setup.py build_ext --inplace"
    )


class ASMWaveSimulation2D:
    def __init__(self, size=256, domain_size=10.0, wave_speed=1.0, dt=0.01):
        self.c_core = c_backend_asm
        self._sim_ptr = self.c_core.create_simulation(size, domain_size, wave_speed, dt)

    def add_wave_source(self, x_pos, y_pos, amplitude=1.0, frequency=3.0, width=0.5):
        self.c_core.add_wave_source(self._sim_ptr, x_pos, y_pos, amplitude, frequency, width)

    def step(self):
        self.c_core.step_simulation(self._sim_ptr)

    def get_intensity(self):
        return self.c_core.get_intensity(self._sim_ptr)

    def get_real_part(self):
        return self.c_core.get_real_part(self._sim_ptr)
