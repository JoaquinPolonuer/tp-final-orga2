from setuptools import setup, Extension
import numpy

# Define the extension module
c_backend_extension = Extension(
    'backends.c_backend_core',
    sources=['backends/c_backend_core.c'],
    include_dirs=[numpy.get_include()],
    libraries=['m'],  # Link math library
    extra_compile_args=['-O3', '-ffast-math', '-march=native'],
    extra_link_args=['-lm']
)

setup(
    name='wave_simulation_backends',
    version='1.0',
    description='C backend for wave simulation',
    ext_modules=[c_backend_extension],
    zip_safe=False,
)