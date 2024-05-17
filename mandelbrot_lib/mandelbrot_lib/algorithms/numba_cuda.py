import numpy as np
from numba import cuda
from ._base import _VectorBaseAlgorithm


@cuda.jit(device=True)
def _compute_escape_iter(x: float, y: float, max_iter: int, escape_radius: float) -> int:
    c = complex(x, y)
    z = 0.0j
    for i in range(max_iter):
        z = z * z + c
        if z.real * z.real + z.imag * z.imag >= escape_radius**2:
            return i
    return -1


@cuda.jit
def compute_grid_kernel(
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    width: int,
    height: int,
    max_iter: int,
    results: np.ndarray,
    escape_radius: float,
):
    """Compute the Mandelbrot set for a given grid on the GPU."""
    pixel_size_x = (x_max - x_min) / width
    pixel_size_y = (y_max - y_min) / height
    x, y = cuda.grid(2)
    if x < width and y < height:
        real = x_min + x * pixel_size_x
        imag = y_min + y * pixel_size_y
        results[y, x] = _compute_escape_iter(real, imag, max_iter, escape_radius)


class NumbaCuda(_VectorBaseAlgorithm):
    """Numba-based Mandelbrot set computation algorithm for GPU."""

    def compute_escape_iter(self, x: float, y: float, max_iter: int) -> int:
        return _compute_escape_iter(x, y, max_iter, self.escape_radius)

    def compute_grid(
        self, x_min: float, y_min: float, x_max: float, y_max: float, width: int, height: int, max_iter: int
    ) -> np.ndarray:
        results = np.zeros((height, width), dtype=np.int32)
        d_results = cuda.to_device(results)

        # Define grid and block dimensions
        nthreads = 16
        blockspergrid_x = (width + nthreads - 1) // nthreads
        blockspergrid_y = (height + nthreads - 1) // nthreads

        compute_grid_kernel[(blockspergrid_x, blockspergrid_y), (nthreads, nthreads)](
            x_min, y_min, x_max, y_max, width, height, max_iter, d_results, self.escape_radius
        )

        # Copy the result back to the host
        d_results.copy_to_host(results)

        return results
