import numpy as np
from ._base import _VectorBaseAlgorithm


class NumpyGrid(_VectorBaseAlgorithm):
    """Numpy-based Mandelbrot set computation algorithm."""

    def compute_escape_iter(self, x: np.ndarray, y: np.ndarray, max_iter: int) -> np.ndarray:
        complex_grid = x + 1j * y
        z = np.zeros_like(complex_grid, dtype=np.complex64)
        escape_iterations = np.full_like(complex_grid, -1, dtype=np.int32)
        diverged = np.zeros_like(complex_grid, dtype=bool)
        for i in range(max_iter):
            z[~diverged] = z[~diverged] * z[~diverged] + complex_grid[~diverged]
            escaped = (z.real * z.real + z.imag * z.imag) >= self.escape_radius**2
            escape_iterations[np.logical_and(escaped, escape_iterations == -1)] = i
            diverged = np.logical_or(diverged, escaped)
        return escape_iterations

    def compute_grid(
        self, x_min: float, y_min: float, x_max: float, y_max: float, width: int, height: int, max_iter: int
    ) -> np.ndarray:
        real = np.linspace(x_min, x_max, width)
        imag = np.linspace(y_min, y_max, height)
        real_grid, imag_grid = np.meshgrid(real, imag)
        results = self.compute_escape_iter(real_grid, imag_grid, max_iter)
        return results
