import numpy as np
from mandelbrot_lib.algorithms.numpy_grid import NumpyGrid
from mandelbrot_lib.settings import DEFAULT_ITER, DEFAULT_ESCAPE_RADIUS, DEFAULT_HEIGHT, DEFAULT_WIDTH


def test_numpy_grid_initialization():
    algo = NumpyGrid()
    assert algo.escape_radius == DEFAULT_ESCAPE_RADIUS


def test_numpy_grid_compute_escape_iter():
    algo = NumpyGrid()
    result = algo.compute_escape_iter(0, 0, DEFAULT_ITER)
    assert result == DEFAULT_ITER


def test_numpy_grid_compute_grid():
    algo = NumpyGrid()
    result = algo.compute_grid(x_min=0, y_min=0, x_max=1, y_max=1, width=10, height=10, max_iter=DEFAULT_ITER)
    assert isinstance(result, np.ndarray)
    assert result.shape == (10, 10)


def test_numpy_grid_benchmark_defaults():
    algo = NumpyGrid()
    result = algo.benchmark_defaults(DEFAULT_ITER)
    assert isinstance(result, float)


def test_numpy_grid_benchmark_compute_grid_defaults():
    algo = NumpyGrid()
    result = algo.benchmark_compute_grid_defaults(DEFAULT_WIDTH, DEFAULT_HEIGHT, DEFAULT_ITER)
    assert isinstance(result, float)

