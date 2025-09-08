from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import subprocess
import os

class CustomBuildExt(build_ext):
    def build_extension(self, ext):
        asm_files = {
            "backends.c_backend_asm": "backends/fft_asm.asm",
        }

        asm_src = asm_files.get(ext.name)
        if asm_src:
            asm_obj = os.path.splitext(asm_src)[0] + ".o"

            # Only (re)compile if needed
            if not os.path.exists(asm_obj) or os.path.getmtime(asm_src) > os.path.getmtime(asm_obj):
                subprocess.check_call(["nasm", "-f", "elf64", asm_src, "-o", asm_obj])

            # Ensure extra_objects exists and add object file
            if not hasattr(ext, "extra_objects") or ext.extra_objects is None:
                ext.extra_objects = []
            if asm_obj not in ext.extra_objects:
                ext.extra_objects.append(asm_obj)

            # Remove .asm from sources to avoid setuptools confusion
            if asm_src in ext.sources:
                ext.sources.remove(asm_src)

        super().build_extension(ext)


c_backend_extension = Extension(
    "backends.c_backend",
    sources=["backends/c_backend.c", "backends/wave_simulation_common.c"],
)

c_backend_asm_extension = Extension(
    "backends.c_backend_asm",
    sources=["backends/c_backend_asm.c", "backends/wave_simulation_common.c", "backends/fft_asm.asm"],
)
c_backend_avx_extension = Extension(
    "backends.c_backend_avx",
    sources=["backends/c_backend_avx.c", "backends/wave_simulation_common.c"],
    extra_compile_args=["-mavx", "-mavx2"],
)

ext_modules = [c_backend_extension, c_backend_asm_extension, c_backend_avx_extension]

setup(
    name="wave_simulation_backends",
    version="1.0",
    description="C backends for wave simulation (NumPy-based and Pure C)",
    ext_modules=ext_modules,
    cmdclass={"build_ext": CustomBuildExt},
    packages=[],
    zip_safe=False,
)
