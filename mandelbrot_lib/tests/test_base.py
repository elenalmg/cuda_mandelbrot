import numpy as np
from mandelbrot_lib.algorithms._base import _ComplexBaseAlgorithm, _VectorBaseAlgorithm
from mandelbrot_lib.settings import DEFAULT_ESCAPE_RADIUS, DEFAULT_ITER


class MockComplexBaseAlgorithm(_ComplexBaseAlgorithm):
    def compute_escape_iter(self, c: complex, max_iter: int = DEFAULT_ITER):
        return max_iter

    def compute_grid(self, c1: complex, c2: complex, width: int, height: int, max_iter: int):
        return np.zeros((height, width), dtype=int)


class MockVectorBaseAlgorithm(_VectorBaseAlgorithm):
    def compute_escape_iter(self, x: float, y: float, max_iter: int = DEFAULT_ITER):
        return max_iter

    def compute_grid(
        self, x_min: float, x_max: float, y_min: float, y_max: float, width: int, height: int, max_iter: int
    ):
        return [[0] * width for _ in range(height)]


def test_complex_base_initialization():
    algo = MockComplexBaseAlgorithm()
    assert algo.escape_radius == DEFAULT_ESCAPE_RADIUS


def test_complex_compute_escape_iter():
    algo = MockComplexBaseAlgorithm()
    result = algo.compute_escape_iter(0 + 0j, DEFAULT_ITER)
    assert result == DEFAULT_ITER


def test_complex_compute_grid():
    algo = MockComplexBaseAlgorithm()
    result = algo.compute_grid(0 + 0j, 1 + 1j, 10, 10, DEFAULT_ITER)
    assert isinstance(result, np.ndarray)
    assert result.shape == (10, 10)


def test_complex_benchmark_defaults():
    algo = MockComplexBaseAlgorithm()
    result = algo.benchmark_defaults([DEFAULT_ITER])
    assert isinstance(result, float)


def test_vector_base_initialization():
    algo = MockVectorBaseAlgorithm()
    assert algo.escape_radius == DEFAULT_ESCAPE_RADIUS


def test_vector_compute_escape_iter():
    algo = MockVectorBaseAlgorithm()
    result = algo.compute_escape_iter(0.0, 0.0, DEFAULT_ITER)
    assert result == DEFAULT_ITER


def test_vector_compute_grid():
    algo = MockVectorBaseAlgorithm()
    result = algo.compute_grid(-1.0, 1.0, -1.0, 1.0, 10, 10, DEFAULT_ITER)
    assert isinstance(result, list)
    assert all(isinstance(row, list) for row in result)
    assert all(len(row) == 10 for row in result)
    assert len(result) == 10


def test_vector_benchmark_defaults():
    algo = MockVectorBaseAlgorithm()
    result = algo.benchmark_defaults([DEFAULT_ITER])
    assert isinstance(result, float)
