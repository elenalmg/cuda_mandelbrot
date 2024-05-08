import numpy as np
from abc import ABC, abstractmethod
from typing import Any
from time import time
from ..settings import BASE_COMPLEX, BASE_VECTORS, BASE_GRID_COMPLEX, BASE_GRID_VECTORS, DEFAULT_ESCAPE_RADIUS, DEFAULT_ITER, DEFAULT_HEIGHT, DEFAULT_WIDTH


class _BaseAlgorithm(ABC):
    def __init__(self, escape_radius: float = DEFAULT_ESCAPE_RADIUS):
        self.escape_radius = escape_radius

    @abstractmethod
    def get_base(self):
        pass
    
    @abstractmethod
    def get_base_grid(self):
        pass

    def benchmark_defaults(self, iterations: int) -> float:
        total_time = 0
        for iter in range(iterations):
            for entity in self.get_base():
                start = time()
                self.compute_escape_iter(*entity, iter) if isinstance(entity, tuple) else self.compute_escape_iter(entity, iter)
                total_time += time() - start
        return total_time

    def benchmark_compute_grid_defaults(self, width : int, height : int, iterations: int) -> float:
        total_time = 0
        for iter in range(iterations):
            for entity in self.get_base_grid():
                start = time()
                self.compute_grid(*entity[0]+entity[1], width, height, iter) if isinstance(entity[0], tuple) else self.compute_grid(*entity, width, height, iter)
                total_time += time() - start
        return total_time

class _VectorBaseAlgorithm(_BaseAlgorithm):
    """Base class for vector-based Mandelbrot set computation algorithms."""

    def __init__(self, escape_radius: float = DEFAULT_ESCAPE_RADIUS):
        super().__init__(escape_radius)

    def get_base(self):
        return BASE_VECTORS
    
    def get_base_grid(self):
        return BASE_GRID_VECTORS

    @abstractmethod
    def compute_escape_iter(self, x: float, y: float, max_iter: int = DEFAULT_ITER) -> Any:
        pass

    @abstractmethod
    def compute_grid(
        self, x_min: float, y_min: float, x_max: float, y_max: float, width: int = DEFAULT_WIDTH, height: int = DEFAULT_HEIGHT, max_iter: int = DEFAULT_ITER) -> list:
        pass


class _ComplexBaseAlgorithm(_BaseAlgorithm):
    """Base class for complex-based Mandelbrot set computation algorithms."""

    def __init__(self, escape_radius: float = DEFAULT_ESCAPE_RADIUS):
        super().__init__(escape_radius)

    def get_base(self):
        return BASE_COMPLEX
    
    def get_base_grid(self):
        return BASE_GRID_COMPLEX

    @abstractmethod
    def compute_escape_iter(self, c: complex, max_iter: int = DEFAULT_ITER) -> Any:
        pass

    @abstractmethod
    def compute_grid(self, c1: complex, c2: complex, width: int = DEFAULT_WIDTH, height: int = DEFAULT_HEIGHT, max_iter: int = DEFAULT_ITER) -> np.ndarray:
        pass
