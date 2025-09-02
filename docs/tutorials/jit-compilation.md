---
sidebar_position: 2
---

# JIT-compiling a Python function

In this tutorial, we walk through how to rewrite a Python function operating on lists to a function that we can JIT-compile using ParPy.

## Matrix-vector multiplication

Consider the following implementation of a matrix-vector multiplication in Python operating on lists:
```python
def mv(A, b):
    return [sum([A_val * b_val for (A_val, b_val) in zip(A_row, b)]) for A_row in A]

A = [[2.5, 3.5], [1.5, 0.5]]
b = [2.0, 1.0]
mv(A, b)
```

The `mv` function, as implemented above, cannot be JIT-compiled using ParPy. We will consider issues, and discuss how the code can be rewritten to eventually make it parallelizable. We also motivate *why* we have to perform certain rewrites.

### Returning a value

A problem with the `mv` function implementation is that it returns the result. In ParPy, we can only return values in helper functions (i.e., JIT-compiled functions that are only called from other JIT-compiled functions). Instead, the function should take the output data as arguments and mutate it. ParyPy is designed in this way because it not allow (explicit) allocations of data within JIT-compiled functions. In turn, this is because we cannot allocate memory inside GPU code and access it outside the GPU kernel (the function executing the GPU code). We rewrite the function by having it mutate a pre-allocated output list (assuming `N` is the number of rows in `A`). Note that we cannot assign the list comprehension expression of the original implementation to `out` immediately, as that would make the `out` inside the function a distinct object (hence, the updates would not be visible outside the function). Instead, we mutate it row-by-row in a for-loop.
```python
def mv(A, b, out, N):
    for row in range(N):
        out[row] = sum([A_val * b_val for (A_val, b_val) in zip(A[row], b)])

N = 2
A = [[2.5, 3.5], [1.5, 0.5]]
b = [2.0, 1.0]
out = [0.0 for i in range(N)]
mv(A, b, out, N)
```

### Python lists

The next problem with the function is that it uses Python lists. These are great to use in Python, but they are difficult to use in native code (e.g., because they can contain elements of different types). For a similar reason, we do not support list comprehensions. However, we can perform elementwise multiplication on NumPy arrays:
```python
import numpy as np

def mv(A, b, out, N):
    for row in range(N):
        out[row] = sum(A[row] * b)

N = 2
A = np.array([[2.5, 3.5], [1.5, 0.5]])
b = np.array([2.0, 1.0])
out = np.zeros((N,))
mv(A, b, out, N)
```

ParPy supports arrays implementing the [Array Interface Protocol](https://numpy.org/doc/stable/reference/arrays.interface.html) and the similar [CUDA Array Interface](https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html). This includes containers such as [NumPy](https://numpy.org/) arrays, [PyTorch](https://pytorch.org/) tensors, and [CuPy](https://cupy.dev/) arrays. ParPy will use the data of a CUDA array (e.g., a PyTorch tensor placed on the GPU) without copying, while other kinds of data (e.g., a NumPy array) is automatically copied between the CPU and the GPU.

### Parallelization

At this stage, we can enable parallelization of the `mv` function by adding a decorator and labels. ParPy also requires us to rewrite the expression inside the summation from `A[row] * b` to `A[row,:] * b[:]`. This is because the latter form explicitly lists the dimension we parallelize over (this is controlled by the associated label `M`). The resulting code looks like:
```python
import parpy

@parpy.jit
def mv(A, b, out, N):
    parpy.label('N')
    for i in range(N):
        parpy.label('M')
        out[row] = sum(A[row, :] * b[:])
```

When calling the JIT-compiled `mv` function, we have to provide the compiler options where we specify how to parallelize it (as we saw in the [previous tutorial](./basic-parallelization.md)):
```
opts = parpy.par({'N': parpy.threads(N)})
mv(A, b, out, N, opts=opts)
```
