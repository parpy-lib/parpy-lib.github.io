---
sidebar_position: 2
---

# JIT-compiling a Python function

In this tutorial, we walk through how to rewrite a Python function operating on lists to a function that we can JIT-compile using ParPy. The code is available at `examples/jit-compilation.py` in the ParPy repository, and can be run using `python examples/jit-compilation.py`.

## Matrix-vector multiplication

Consider the following implementation of a matrix-vector multiplication in Python operating on lists:
```python
def mv(A, b):
    return [sum([A_val * b_val for (A_val, b_val) in zip(A_row, b)]) for A_row in A]

A = [[2.5, 3.5], [1.5, 0.5]]
b = [2.0, 1.0]
out = mv(A, b)
print("lists", out)
```

The `mv` function, as implemented above, cannot be JIT-compiled using ParPy. We will consider issues, and discuss how the code can be rewritten to eventually make it parallelizable. We also motivate *why* we have to perform certain rewrites.

### Returning a value

A problem with the `mv` function implementation is that it returns the result. In ParPy, we can only return values in helper functions (i.e., JIT-compiled functions that are only called from other JIT-compiled functions). Instead, the function should take the output data as arguments and mutate it. ParyPy is designed in this way because it not allow (explicit) allocations of data within JIT-compiled functions. In turn, this is because we cannot allocate memory inside GPU code and access it outside the GPU kernel (the function executing the GPU code). We rewrite the function by having it mutate a pre-allocated output list (assuming `N` is the number of rows in `A`). Note that we cannot assign the list comprehension expression of the original implementation to `out` immediately, as that would make the `out` inside the function a distinct object (hence, the updates would not be visible outside the function). Instead, we mutate it row-by-row in a for-loop.
```python
def mv_inplace(A, b, out, N):
    for row in range(N):
        out[row] = sum([A_val * b_val for (A_val, b_val) in zip(A[row], b)])

N = 2
A = [[2.5, 3.5], [1.5, 0.5]]
b = [2.0, 1.0]
out = [0.0 for i in range(N)]
mv_inplace(A, b, out, N)
print("lists (no return)", out)
```

### Python lists

The next problem with the function is that it uses Python lists. These are great to use in Python, but they are difficult to use in native code (e.g., because they can contain elements of different types). For a similar reason, we do not support list comprehensions. However, we can perform elementwise multiplication on NumPy arrays:
```python
import numpy as np

def mv_numpy(A, b, out, N):
    for row in range(N):
        out[row] = sum(A[row] * b)

N = 2
M = 2
A = np.array([[2.5, 3.5], [1.5, 0.5]])
b = np.array([2.0, 1.0])
out = np.zeros((N,))
mv_numpy(A, b, out, N)
print("NumPy v2", out)
```

ParPy supports arrays implementing the [Array Interface Protocol](https://numpy.org/doc/stable/reference/arrays.interface.html) and the similar [CUDA Array Interface](https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html). This includes containers such as [NumPy](https://numpy.org/) arrays, [PyTorch](https://pytorch.org/) tensors, and [CuPy](https://cupy.dev/) arrays. ParPy will use the data of a CUDA array (e.g., a PyTorch tensor placed on the GPU) without copying, while other kinds of data (e.g., a NumPy array) is automatically copied between the CPU and the GPU.

### Parallelization

At this stage, we can enable parallelization of the function by adding a decorator and labels. ParPy also requires us to rewrite the expression inside the summation from `A[row] * b` to `A[row,:] * b[:]`. This is because the latter form explicitly lists the dimension we parallelize over (this is controlled by the associated label `M`). The resulting code is:
```python
import parpy

@parpy.jit
def mv_parpy(A, b, out, N):
    parpy.label('N')
    for i in range(N):
        parpy.label('M')
        out[row] = sum(A[row, :] * b[:])
```

When calling the JIT-compiled `mv_parpy` function, we have to provide the compiler options where we specify how to parallelize it (as we saw in the [previous tutorial](./basic-parallelization.md)):
```
opts = parpy.par({'N': parpy.threads(N)})
mv_parpy(A, b, out, N, opts=opts)
print("ParPy", out)
```

In this example, it is not useful to use ParPy, as the input sizes are extremely small. Generally, if we have more data, we can gain more from parallelizing it on the GPU. However, in this example we perform little computational work relative to the amount of memory we allocate. As the arguments provided to `mv_parpy` are NumPy arrays, ParPy will implicitly copy data back and forth between the CPU (where NumPy arrays are stored) and the GPU.

## Using buffers to pre-allocate GPU data

The [parpy.buffer](/docs/reference/buffer) module provides functionality for pre-allocating data on the GPU. This can be used to minimize the need to copy data between the CPU and the GPU, and can be useful for benchmarking purposes (where we are typically interested in the computational time). Assuming we use the CUDA backend, we can pre-allocate data as:
```python
backend = parpy.CompileBackend.Cuda
A = parpy.buffer.from_array(A, backend)
b = parpy.buffer.from_array(b, backend)
out = parpy.buffer.from_array(out, backend)
```

To run this code on Metal, we only have to modify the code by setting `backend` to `parpy.CompileBackend.Metal` instead. After allocating data in this way, we can call the `mv_parpy` function as before. When we want to move data back to the CPU, we use `out.numpy()` to get a NumPy array containing the contents of the `out` array. Note that we have to use `out.sync()` before the end of the timing measurements, to make the CPU wait for the GPU execution to finish. Otherwise, this delay takes place in the `out.numpy()` call. When using NumPy arrays as arguments, this synchronization happens automatically when ParPy copies data back from the GPU.
```python
import time
t1 = time.time_ns()
mv_parpy(A, b, out, N, opts=opts)
out.sync()
t2 = time.time_ns()
print("Time:", (t2-t1)/1e9)
print("ParPy", out.numpy())
```

Try allocating larger amounts of data using `np.random.randn(N, M)` and see how the performance scales for the different versions.
