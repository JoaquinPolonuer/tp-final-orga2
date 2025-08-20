from setuptools import setup, Extension
import numpy

# Define the pure C extension module (no NumPy dependency)
pure_c_backend_extension = Extension(
    "backends.pure_c_backend_core",
    sources=["backends/pure_c_backend_core.c"],
    libraries=["m"],  # Link math library
    extra_compile_args=["-O3", "-ffast-math", "-march=native"],
    extra_link_args=["-lm"],
)

setup(
    name="wave_simulation_backends",
    version="1.0",
    description="C backends for wave simulation (NumPy-based and Pure C)",
    ext_modules=[pure_c_backend_extension],
    zip_safe=False,
)
