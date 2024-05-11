from matplotlib.pyplot import imshow, show
from timeit import default_timer as timer
import numpy as np
from numba import cuda
from numba import *
from ._base import _VectorBaseAlgorithm


class NumbaCuda(_VectorBaseAlgorithm):
    """Numba-based Mandelbrot set computation algorithm for GPU."""

    @cuda.jit(device=True)
    def compute_escape_iter(self, x : float, y : float, max_iter: int) -> np.ndarray:
        c = complex(x, y)
        z = 0.0j
        for i in range(max_iter):
            z = z * z + c
            if z.real * z.real + z.imag * z.imag >= self.escape_radius**2:
                return i
        return max_iter
    
    
    def compute_grid(self, x_min: float, y_min: float, x_max: float, y_max: float, width: int, height: int, max_iter: int) -> np.ndarray:
        results = np.zeros((height, width), dtype=np.int32)

        # Define grid and block dimensions
        nthreads = 16
        blockspergrid_x = (width + nthreads - 1) // nthreads
        blockspergrid_y = (height + nthreads - 1) // nthreads

        @cuda.jit
        def compute_grid_kernel(self, x_min: float, y_min: float, x_max: float, y_max: float, width: int, height: int, max_iter: int, results: np.ndarray):
            pixel_size_x = (x_max - x_min) / width
            pixel_size_y = (y_max - y_min) / height
            x, y = cuda.grid(2)
            if x < width and y < height:
                real = x_min + x * pixel_size_x
                imag = y_min + y * pixel_size_y
                results[y, x] = self.compute_escape_iter(real, imag, max_iter)

        compute_grid_kernel[(blockspergrid_x, blockspergrid_y) , (nthreads, nthreads)](x_min, y_min, x_max, y_max, width, height, max_iter, results)
        return results




