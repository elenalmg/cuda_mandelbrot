import numpy as np
from numba import cuda
from ._base import _VectorBaseAlgorithm


@cuda.jit(device=True)
def _compute_escape_iter(real: float, imag: float, max_iter: int, escape_radius_squared: float) -> int:
    zr = 0.0
    zi = 0.0
    for i in range(max_iter):
        zr2 = zr * zr
        zi2 = zi * zi
        if zr2 + zi2 >= escape_radius_squared:
            return i
        zi = 2.0 * zr * zi + imag
        zr = zr2 - zi2 + real
    return -1


@cuda.jit
def _compute_grid_kernel(
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    width: int,
    height: int,
    max_iter: int,
    results: np.ndarray,
    escape_radius_squared: float,
):
    pixel_size_x = (x_max - x_min) / width
    pixel_size_y = (y_max - y_min) / height
    x, y = cuda.grid(2)
    if x < width and y < height:
        real = x_min + x * pixel_size_x
        imag = y_min + y * pixel_size_y
        results[y, x] = _compute_escape_iter(real, imag, max_iter, escape_radius_squared)


class NumbaCuda(_VectorBaseAlgorithm):
    """Numba-based Mandelbrot set computation algorithm for GPU."""

    def compute_escape_iter(self, x: float, y: float, max_iter: int) -> int:
        return _compute_escape_iter(x, y, max_iter, self.escape_radius**2)

    def compute_grid(
        self, x_min: float, y_min: float, x_max: float, y_max: float, width: int, height: int, max_iter: int
    ) -> np.ndarray:
        results = np.zeros((height, width), dtype=np.int32)
        d_results = cuda.to_device(results)

        # Define grid and block dimensions
        nthreads = 16
        blockspergrid_x = (width + nthreads - 1) // nthreads
        blockspergrid_y = (height + nthreads - 1) // nthreads

        _compute_grid_kernel[(blockspergrid_x, blockspergrid_y), (nthreads, nthreads)](
            x_min, y_min, x_max, y_max, width, height, max_iter, d_results, self.escape_radius**2
        )

        # Copy the result back to the host
        d_results.copy_to_host(results)

        return results
