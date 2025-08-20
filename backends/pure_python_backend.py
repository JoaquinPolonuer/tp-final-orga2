import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from abc import ABC, abstractmethod
import math
import cmath
import math
import cmath
from typing import Union

class PurePythonBackend:
    """Complete Pure Python backend implementation with proper FFT"""

    def zeros(self, shape: Union[int, tuple[int, int]], dtype=complex) -> Union[list, list[list]]:
        if isinstance(shape, int):
            return [0 + 0j if dtype == complex else 0] * shape
        rows, cols = shape
        return [[0 + 0j if dtype == complex else 0 for _ in range(cols)] for _ in range(rows)]

    def linspace(self, start: float, stop: float, num: int) -> list[float]:
        if num == 1:
            return [start]
        step = (stop - start) / (num - 1)
        return [start + i * step for i in range(num)]

    def meshgrid(
        self, x: list[float], y: list[float]
    ) -> tuple[list[list[float]], list[list[float]]]:
        X = [[x[j] for j in range(len(x))] for i in range(len(y))]
        Y = [[y[i] for j in range(len(x))] for i in range(len(y))]
        return X, Y

    def sqrt(self, x: Union[float, complex, list]) -> Union[float, list]:
        if isinstance(x, (list, tuple)):
            result = []
            for i in range(len(x)):
                if isinstance(x[i], (list, tuple)):
                    row = []
                    for j in range(len(x[i])):
                        val = x[i][j]
                        if isinstance(val, complex):
                            row.append(math.sqrt(val.real**2 + val.imag**2))
                        else:
                            row.append(math.sqrt(abs(val)))
                    result.append(row)
                else:
                    val = x[i]
                    if isinstance(val, complex):
                        result.append(math.sqrt(val.real**2 + val.imag**2))
                    else:
                        result.append(math.sqrt(abs(val)))
            return result

        if isinstance(x, complex):
            return math.sqrt(x.real**2 + x.imag**2)
        return math.sqrt(abs(x))

    def exp(self, x: Union[float, complex, list]) -> Union[float, complex, list]:
        if isinstance(x, (list, tuple)):
            result = []
            for i in range(len(x)):
                if isinstance(x[i], (list, tuple)):
                    row = []
                    for j in range(len(x[i])):
                        val = x[i][j]
                        if isinstance(val, complex):
                            row.append(cmath.exp(val))
                        else:
                            row.append(math.exp(val))
                    result.append(row)
                else:
                    val = x[i]
                    if isinstance(val, complex):
                        result.append(cmath.exp(val))
                    else:
                        result.append(math.exp(val))
            return result

        if isinstance(x, complex):
            return cmath.exp(x)
        return math.exp(x)

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

    def abs(self, x: Union[complex, list]) -> Union[float, list]:
        """Absolute value / magnitude"""
        if isinstance(x, (list, tuple)):
            result = []
            for i in range(len(x)):
                if isinstance(x[i], (list, tuple)):
                    row = []
                    for j in range(len(x[i])):
                        val = x[i][j]
                        if isinstance(val, complex):
                            row.append(math.sqrt(val.real**2 + val.imag**2))
                        else:
                            row.append(abs(val))
                    result.append(row)
                else:
                    val = x[i]
                    if isinstance(val, complex):
                        result.append(math.sqrt(val.real**2 + val.imag**2))
                    else:
                        result.append(abs(val))
            return result

        if isinstance(x, complex):
            return math.sqrt(x.real**2 + x.imag**2)
        return abs(x)

    def real(self, x: Union[complex, list]) -> Union[float, list]:
        """Real part extraction"""
        if isinstance(x, (list, tuple)):
            result = []
            for i in range(len(x)):
                if isinstance(x[i], (list, tuple)):
                    row = []
                    for j in range(len(x[i])):
                        val = x[i][j]
                        if isinstance(val, complex):
                            row.append(val.real)
                        else:
                            row.append(val)
                    result.append(row)
                else:
                    val = x[i]
                    if isinstance(val, complex):
                        result.append(val.real)
                    else:
                        result.append(val)
            return result

        if isinstance(x, complex):
            return x.real
        return x

    def max(self, x: Union[float, list]) -> float:
        """Maximum value"""
        if isinstance(x, (list, tuple)):
            if isinstance(x[0], (list, tuple)):
                # 2D array
                max_val = float("-inf")
                for row in x:
                    for val in row:
                        if val > max_val:
                            max_val = val
                return max_val
            else:
                # 1D array
                return max(x)
        return x

    def sum(self, x: Union[float, list]) -> float:
        """Sum of all elements"""
        if isinstance(x, (list, tuple)):
            if isinstance(x[0], (list, tuple)):
                # 2D array
                total = 0
                for row in x:
                    for val in row:
                        total += val
                return total
            else:
                # 1D array
                return sum(x)
        return x

    def clip(self, x: float, min_val: float, max_val: float) -> float:
        """Clip value to range"""
        return max(min_val, min(max_val, x))

    def multiply_elementwise(self, a: list, b: list) -> list:
        """Element-wise multiplication for arrays"""
        if isinstance(a[0], (list, tuple)):
            # 2D arrays
            result = []
            for i in range(len(a)):
                row = []
                for j in range(len(a[i])):
                    row.append(a[i][j] * b[i][j])
                result.append(row)
            return result
        else:
            # 1D arrays
            return [a[i] * b[i] for i in range(len(a))]

    def add_elementwise(self, a: list, b: list) -> list:
        """Element-wise addition for arrays"""
        if isinstance(a[0], (list, tuple)):
            # 2D arrays
            result = []
            for i in range(len(a)):
                row = []
                for j in range(len(a[i])):
                    row.append(a[i][j] + b[i][j])
                result.append(row)
            return result
        else:
            # 1D arrays
            return [a[i] + b[i] for i in range(len(a))]

    def power(self, x: Union[float, list], p: float) -> Union[float, list]:
        """Element-wise power operation"""
        if isinstance(x, (list, tuple)):
            result = []
            for i in range(len(x)):
                if isinstance(x[i], (list, tuple)):
                    row = []
                    for j in range(len(x[i])):
                        row.append(x[i][j] ** p)
                    result.append(row)
                else:
                    result.append(x[i] ** p)
            return result
        return x**p
