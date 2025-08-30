from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import subprocess
import os


class CustomBuildExt(build_ext):
    def build_extension(self, ext):
        if ext.name == "backends.c_backend_asm":
            # Compile assembly file to object file
            asm_src = "backends/fft_asm.asm"
            asm_obj = "backends/fft_asm.o"

            if not os.path.exists(asm_obj) or os.path.getmtime(asm_src) > os.path.getmtime(asm_obj):
                subprocess.check_call(["nasm", "-f", "elf64", asm_src, "-o", asm_obj])

            # Add the object file to extra link args
            if asm_obj not in ext.extra_objects:
                ext.extra_objects.append(asm_obj)

            # Remove the .asm file from sources to avoid setuptools confusion
            if asm_src in ext.sources:
                ext.sources.remove(asm_src)

        elif ext.name == "backends.c_backend_asm_avx":
            # Compile AVX assembly file to object file
            asm_src = "backends/fft_asm_avx.asm"
            asm_obj = "backends/fft_asm_avx.o"

            if not os.path.exists(asm_obj) or os.path.getmtime(asm_src) > os.path.getmtime(asm_obj):
                subprocess.check_call(["nasm", "-f", "elf64", asm_src, "-o", asm_obj])

            # Add the object file to extra link args
            if asm_obj not in ext.extra_objects:
                ext.extra_objects.append(asm_obj)

            # Remove the .asm file from sources to avoid setuptools confusion
            if asm_src in ext.sources:
                ext.sources.remove(asm_src)

        super().build_extension(ext)


c_backend_extension = Extension(
    "backends.c_backend",
    sources=["backends/c_backend.c"],
)

c_backend_asm_extension = Extension(
    "backends.c_backend_asm",
    sources=["backends/c_backend_asm.c", "backends/fft_asm.asm"],
)

c_backend_asm_avx = Extension(
    "backends.c_backend_asm_avx",
    sources=["backends/c_backend_asm_avx.c", "backends/fft_asm_avx.asm"],
    extra_compile_args=["-mavx", "-mavx2"],
)

c_backend_optimized_extension = Extension(
    "backends.c_backend_optimized",
    sources=["backends/c_backend_optimized.c"],
    extra_compile_args=["-mavx", "-mavx2"],
)

ext_modules = [c_backend_extension, c_backend_asm_extension, c_backend_asm_avx, c_backend_optimized_extension]

setup(
    name="wave_simulation_backends",
    version="1.0",
    description="C backends for wave simulation (NumPy-based and Pure C)",
    ext_modules=ext_modules,
    cmdclass={"build_ext": CustomBuildExt},
    zip_safe=False,
)
