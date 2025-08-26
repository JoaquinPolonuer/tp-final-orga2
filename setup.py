from setuptools import setup, Extension

c_backend_extension = Extension(
    "backends.c_backend",
    sources=["backends/c_backend.c"],
    libraries=["m"],  # Link math library
    extra_compile_args=["-O3", "-ffast-math"],
    extra_link_args=["-lm"],
)

c_backend_optimized_extension = Extension(
    "backends.c_backend_optimized",
    sources=["backends/c_backend_optimized.c"],
    libraries=["m"],  # Link math library
    extra_compile_args=["-O3", "-ffast-math", "-mavx", "-mavx2"],
    extra_link_args=["-lm"],
)


setup(
    name="wave_simulation_backends",
    version="1.0",
    description="C backends for wave simulation (NumPy-based and Pure C)",
    ext_modules=[c_backend_extension, c_backend_optimized_extension],
    zip_safe=False,
)
