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

- Number of threads per block: 16x16 (256 threads).
- Number of blocks: Determined by the grid size divided by the number of threads per block.

```python
numba = NumbaCuda(escape_radius=escape_radius)
```

### 4. C++ Implementation

This is a C++ implementation of the Mandelbrot set computation. It provides a performance improvement over Python by using lower-level operations and optimizations available in C++. This implementation uses multithreading to parallelize the computation across multiple CPU cores.

```python
cpp = MandelbrotCPP(escape_radius=escape_radius)
```

### 5. CUDA Implementation

The CUDA implementation leverages NVIDIA's CUDA platform to perform parallel computation on the GPU. CUDA allows for massive parallelism by using thousands of lightweight threads. This implementation uses:

- Number of threads per block: 16x16 (256 threads).
- Number of blocks: Determined by the grid size divided by the number of threads per block.

```python
cuda = MandelbrotCUDA(escape_radius=escape_radius)
```

### 6. Fast Mandelbrot Algorithm

The Fast Mandelbrot algorithm is our attempt at a more optimized algorithm. It reduces the number of iterations required by performing two iterations in a single step.

**Single Iteration**

A single iteration of the Mandelbrot function is:

\[ z\_{n+1} = z_n^2 + c \]

Expanding this using vector notation, we get:

\[ z*{n+1} = \begin{pmatrix} x \\ y \end{pmatrix}^2 + \begin{pmatrix} a \\ b \end{pmatrix} \]
\[ z*{n+1} = \begin{pmatrix} x^2 - y^2 \\ 2xy \end{pmatrix} + \begin{pmatrix} a \\ b \end{pmatrix} \]
\[ z\_{n+1} = \begin{pmatrix} x^2 - y^2 + a \\ 2xy + b \end{pmatrix} \]

**Double iteration**

Combining two iterations into one, we get:

\[ z*{n+2} = (z*{n+1})^2 + c = ((z_n^2 + c)^2 + c) \]

Let \( \begin{pmatrix} x_1 \\ y_1 \end{pmatrix} = \begin{pmatrix} x^2 - y^2 + a \\ 2xy + b \end{pmatrix} \), then:

\[ z*{n+2} = \begin{pmatrix} x_1 \\ y_1 \end{pmatrix}^2 + \begin{pmatrix} a \\ b \end{pmatrix} \]
\[ z*{n+2} = \begin{pmatrix} x*1^2 - y_1^2 \\ 2x_1y_1 \end{pmatrix} + \begin{pmatrix} a \\ b \end{pmatrix} \]
\[ z*{n+2} = \begin{pmatrix} x_1^2 - y_1^2 + a \\ 2x_1y_1 + b \end{pmatrix} \]

Expanding the terms, we get:

For the real part:
\[ \text{Real part} = (x^2 - y^2 + a)^2 - (2xy + b)^2 + a \]
\[ = (x^2 - y^2 + a)(x^2 - y^2 + a) - (2xy + b)(2xy + b) + a \]
\[ = x^4 - 2x^2y^2 + y^4 + 2ax^2 - 2ay^2 + a^2 - 4x^2y^2 - 4xyb - b^2 + a \]
\[ = x^4 - 6x^2y^2 + y^4 + 2ax^2 - 2ay^2 + a^2 - 4xyb - b^2 + a \]

For the imaginary part:
\[ \text{Imaginary part} = 2(x^2 - y^2 + a)(2xy + b) + b \]
\[ = 2x^2 \cdot 2xy + 2x^2 \cdot b - 2y^2 \cdot 2xy - 2y^2 \cdot b + 2a \cdot 2xy + 2a \cdot b + b \]
\[ = 4x^3y + 2x^2b - 4xy^3 - 2y^2b + 4axy + 2ab + b \]

Therefore:
\[ z\_{n+2} = \begin{pmatrix} x^4 - 6x^2y^2 + y^4 + 2ax^2 - 2ay^2 + a^2 - 4xyb - b^2 + a \\ 4x^3y + 2x^2b - 4xy^3 - 2y^2b + 4axy + 2ab + b \end{pmatrix} \]

### Advantages

By combining two iterations into one, the Fast Mandelbrot algorithm effectively reduces the computational workload. This optimization is particularly beneficial for large-scale computations, such as those required to render high-resolution images of the Mandelbrot set.

### Conclusion

The Fast Mandelbrot algorithm leverages the mathematical properties of complex numbers to perform two iterations in a single step. This optimization significantly improves the performance of Mandelbrot set computations, making it a valuable approach for applications requiring high computational efficiency.
