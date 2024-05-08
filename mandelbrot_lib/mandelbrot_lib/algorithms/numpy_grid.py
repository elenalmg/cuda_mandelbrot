import numpy as np
from ._base import _VectorBaseAlgorithm

class NumpyGrid(_VectorBaseAlgorithm):
    """Numpy-based Mandelbrot set computation algorithm."""

    def compute_escape_iter(self, x: float, y: float, max_iter: int) -> int:
        c = x + 1j * y
        z = 0
        for i in range(max_iter):
            z = z * z + c
            if (z.real * z.real + z.imag * z.imag) >= self.escape_radius**2:
                return i
        return max_iter

    def compute_grid(self, x_min: float, y_min: float, x_max: float, y_max: float, width: int, height: int, max_iter: int) -> list:
        x = np.linspace(x_min, x_max, width)
        y = np.linspace(y_min, y_max, height)
        results = np.zeros((height, width))
        for i in range(width):
            for j in range(height):
                results[i, j] = self.compute_escape_iter(x[i], y[j], max_iter)
        return results




