# CUDA Mandelbrot

Starting point for GPU accelerated python libraries

Adapted from the structure of https://github.com/pkestene/pybind11-cuda

# Mandelbrot Set Computation

This project demonstrates various algorithms for computing the Mandelbrot set. The Mandelbrot set is a complex set of points that produces a fractal shape when plotted. It is defined by iterating the function:

\[ z\_{n+1} = z_n^2 + c \]

where \( c \in \mathbb{C} \) and \( z_0 = 0 \).

As $n$ approaches infinity, the magnitude of \( z_n \) will either remain bounded or grow without bound. Points that remain bounded are considered part of the Mandelbrot set, while points that escape to infinity are not.

<div style="text-align: center;">
  <img src="images/MandelbrotSet.png" alt="Mandelbrot Set" width="500">
  <p><em>Figure 1: Mandelbrot set in the complex plane</em></p>
</div>

## Algorithms

We have implemented various algorithms to compute the Mandelbrot set through a Python package called `mandelbrot_lib`and a C++/CUDA-based Python package called `cuda_mandelbrot_lib`.

```python
from mandelbrot_lib import NaiveSequential, NumpyGrid, NumbaCuda
from cuda_mandelbrot_lib import MandelbrotCPP, MandelbrotCUDA, FastMandelbrotCUDA
```

### 1. Naive Sequential (Python)

This is a straightforward implementation of the Mandelbrot set computation in Python. It iterates over each point in the grid and computes the number of iterations required for the point to escape.

All our implementations are subclasses of a common case class `_BaseAlgorithm`, and offer a similar interface, like the following:

```python
naive = NaiveSequential(escape_radius=escape_radius)
output = naive.compute_grid(xmin, ymin, xmax, ymax, width, height, n_iterations)
```

### 2. Parallel Numpy (Python)

This implementation uses Numpy to parallelize the computation. By leveraging Numpy's vectorized operations, the algorithm can compute multiple points simultaneously, resulting in faster computation times.

```python
numpy = NumpyGrid(escape_radius=escape_radius)
```

### 3. Numba Python (GPU Python)

The Numba implementation utilizes the GPU to accelerate the computation. Numba is a just-in-time compiler for Python that translates Python functions to optimized machine code at runtime. This implementation uses the following configuration:

- Number of blocks: Determined by the grid size.
- Number of threads per block: Typically 16x16 (256 threads) to ensure efficient use of the GPU resources.

### 4. C++ Implementation

This is a C++ implementation of the Mandelbrot set computation. It provides a performance improvement over Python by using lower-level operations and optimizations available in C++. This implementation uses multithreading to parallelize the computation across multiple CPU cores.

### 5. CUDA Implementation

The CUDA implementation leverages NVIDIA's CUDA platform to perform parallel computation on the GPU. CUDA allows for massive parallelism by using thousands of lightweight threads. This implementation uses:

- Number of threads per block: 16x16 (256 threads).
- Number of blocks: Determined by the grid size divided by the number of threads per block.

### 6. FastMandelbrot Implementation

The FastMandelbrot implementation optimizes the computation by performing two iterations in a single step. This reduces the number of iterations required and speeds up the computation. The mathematical transformation for two iterations is given by:

\[ z*{n+1} = (z_n^2 + c) \]
\[ z*{n+2} = (z\_{n+1}^2 + c) \]

This can be expanded and optimized to reduce the number of arithmetic operations, providing a significant performance boost.
