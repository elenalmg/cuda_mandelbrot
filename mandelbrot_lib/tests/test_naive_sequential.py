import numpy as np
from mandelbrot_lib.algorithms.naive_sequential import NaiveSequential
from mandelbrot_lib.settings import DEFAULT_ITER, DEFAULT_ESCAPE_RADIUS, DEFAULT_HEIGHT, DEFAULT_WIDTH


def test_naive_sequential_initialization():
    algo = NaiveSequential()
    assert algo.escape_radius == DEFAULT_ESCAPE_RADIUS


def test_naive_sequential_compute_escape_iter():
    algo = NaiveSequential()
    result = algo.compute_escape_iter(0 + 0j, DEFAULT_ITER)
    assert result == DEFAULT_ITER


def test_naive_sequential_compute_grid():
    algo = NaiveSequential()
    result = algo.compute_grid(0 + 0j, 1 + 1j, 10, 10, DEFAULT_ITER)
    assert isinstance(result, np.ndarray)
    assert result.shape == (10, 10)


def test_naive_sequential_benchmark_defaults():
    algo = NaiveSequential()
    result = algo.benchmark_defaults(DEFAULT_ITER)
    assert isinstance(result, float)


def test_naive_sequential_benchmark_compute_grid_defaults():
    algo = NaiveSequential()
    result = algo.benchmark_compute_grid_defaults(DEFAULT_WIDTH, DEFAULT_HEIGHT, DEFAULT_ITER)
    assert isinstance(result, float)

