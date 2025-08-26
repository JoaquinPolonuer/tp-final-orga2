from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import subprocess
import os

class CustomBuildExt(build_ext):
    def run(self):
        # Pre-assemble .s and .asm files to .o files
        for ext in self.extensions:
            new_sources = []
            for source in ext.sources:
                if source.endswith('.s') or source.endswith('.asm'):
                    if source.endswith('.s'):
                        obj_file = source.replace('.s', '.o')
                        assembler_cmd = ['as', '-64', source, '-o', obj_file]
                    else:  # .asm file
                        obj_file = source.replace('.asm', '.o')
                        assembler_cmd = ['nasm', '-f', 'elf64', source, '-o', obj_file]
                    
                    if not os.path.exists(obj_file) or os.path.getmtime(source) > os.path.getmtime(obj_file):
                        print(f"Assembling {source} to {obj_file}")
                        subprocess.run(assembler_cmd, check=True)
                    new_sources.append(obj_file)
                else:
                    new_sources.append(source)
            ext.sources = new_sources
            ext.extra_objects = [s for s in ext.sources if s.endswith('.o')]
            ext.sources = [s for s in ext.sources if not s.endswith('.o')]
        
        super().run()

# Define the pure C extension module (no NumPy dependency)
c_backend_extension = Extension(
    "backends.c_backend_core",
    sources=["backends/c_backend_core.c"],
    # libraries=["m"],  # Link math library
    # extra_compile_args=["-O3", "-ffast-math", "-march=native"],
    # extra_link_args=["-lm"],
)

# asm_backend_extension = Extension(
#     "backends.asm_backend_core",
#     sources=["backends/asm_backend_core.c", "backends/complex_asm.asm"],
#     # libraries=["m"],  # Link math library
#     # extra_compile_args=["-O3", "-ffast-math", "-march=native"],
#     # extra_link_args=["-lm"],
# )

setup(
    name="wave_simulation_backends",
    version="1.0",
    description="C backends for wave simulation (NumPy-based and Pure C)",
    ext_modules=[c_backend_extension], #, asm_backend_extension],
    # cmdclass={'build_ext': CustomBuildExt},
    zip_safe=False,
)
