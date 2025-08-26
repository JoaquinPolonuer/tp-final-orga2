import numpy as np
import math
import cmath
from backends.backend import Backend


class PurePythonBackend(Backend):
    def _fft_1d(self, x: list[complex]) -> list[complex]:
        """1D FFT using Cooley-Tukey algorithm"""
        n = len(x)
        if n <= 1:
            return x[:]

        # Pad to power of 2 if needed
        if n & (n - 1) != 0:
            next_power = 1 << (n - 1).bit_length()
            x = x + [0 + 0j] * (next_power - n)
            n = next_power

        # Bit-reversal permutation
        j = 0
        for i in range(1, n):
            bit = n >> 1
            while j & bit:
                j ^= bit
                bit >>= 1
            j ^= bit
            if i < j:
                x[i], x[j] = x[j], x[i]

        # Cooley-Tukey FFT
        length = 2
        while length <= n:
            w = cmath.exp(-2j * math.pi / length)
            for i in range(0, n, length):
                wn = 1 + 0j
                for j in range(length // 2):
                    u = x[i + j]
                    v = x[i + j + length // 2] * wn
                    x[i + j] = u + v
                    x[i + j + length // 2] = u - v
                    wn *= w
            length <<= 1

        return x

    def _ifft_1d(self, x: list[complex]) -> list[complex]:
        """1D IFFT using FFT"""
        n = len(x)
        # Conjugate input
        x_conj = [val.conjugate() for val in x]
        # Apply FFT
        result = self._fft_1d(x_conj)
        # Conjugate output and normalize
        return [val.conjugate() / n for val in result]

    def fft2(self, x: list[list[complex]]) -> list[list[complex]]:
        """2D FFT using row-column method"""
        rows = len(x)
        cols = len(x[0]) if rows > 0 else 0

        # FFT on rows
        row_fft = []
        for i in range(rows):
            row_fft.append(self._fft_1d(x[i][:]))

        # FFT on columns
        result = []
        for i in range(rows):
            result.append([0 + 0j] * cols)

        for j in range(cols):
            col = [row_fft[i][j] for i in range(rows)]
            col_fft = self._fft_1d(col)
            for i in range(rows):
                result[i][j] = col_fft[i]

        return result

    def ifft2(self, x: list[list[complex]]) -> list[list[complex]]:
        """2D IFFT using row-column method"""
        rows = len(x)
        cols = len(x[0]) if rows > 0 else 0

        # IFFT on rows
        row_ifft = []
        for i in range(rows):
            row_ifft.append(self._ifft_1d(x[i][:]))

        # IFFT on columns
        result = []
        for i in range(rows):
            result.append([0 + 0j] * cols)

        for j in range(cols):
            col = [row_ifft[i][j] for i in range(rows)]
            col_ifft = self._ifft_1d(col)
            for i in range(rows):
                result[i][j] = col_ifft[i]

        return result

    def fftfreq(self, n: int, d: float = 1.0) -> list[float]:
        """Generate frequency array for FFT"""
        result = []
        for i in range(n):
            if i <= n // 2:
                result.append(i / (n * d))
            else:
                result.append((i - n) / (n * d))
        return result
