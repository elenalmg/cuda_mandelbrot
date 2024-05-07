import numpy as np
from ._base import _ComplexBaseAlgorithm


class NaiveSequential(_ComplexBaseAlgorithm):
    """Naive sequential implementation of the Mandelbrot set computation algorithm."""

    def compute_escape_iter(self, c: complex, max_iter: int) -> float:
        z = 0.0j
        for i in range(max_iter):
            z = z * z + c
            if (z.real * z.real + z.imag * z.imag) >= self.escape_radius**2:
                return i
        return max_iter

    def compute_grid(self, c1: complex, c2: complex, width: int, height: int, max_iter: int) -> np.ndarray:
        result = np.zeros((height, width))
        for y in range(height):
            for x in range(width):
                c = c1 + (c2 - c1) * (x + y * 1j) / (width + height * 1j)
                result[y, x] = self.compute_escape_iter(c, max_iter)
        return result
