# Assignments outside parallel code

Consider the following code performing a parallel summation of the values of a provided array `x` and storing the results in the first index of an output array `y`:
```python
import parpy
@parpy.jit
def sum_values(x, y, N):
    y[0] = 0.0
    parpy.label('N')
    for i in range(N):
        y[0] += x[i]
```

We use the following code to initialize the input and specify the parallelization. In particular, we explicitly specify that the for-loop performs a reduction (the ParPy compiler does not automatically identify this). Also, we specify that the ParPy compiler should generate code for the CUDA backend.
```python
import numpy as np
N = 100
x = np.random.randn(N).astype(np.float32)
y = np.empty((1,), dtype=np.float32)
opts = parpy.par({'N': parpy.threads(N).reduce()})
opts.backend = parpy.CompileBackend.Cuda
print(parpy.print_compiled(sum_values, [x, y, N], opts))
```

When running this code, the ParPy compiler fails and produces an error of the form
```
RuntimeError: Assignments are not allowed outside parallel code.

On line 4 of file /<path>/<to>/<file>/example.py:
    y[0] = 0.0
    ^^^^
```

## Problem

The problem is that the initial statement is placed outside parallel code. Such statements are executed on the CPU. When we use the CUDA backend, we cannot assign to `y` on the CPU because data is allocated on the GPU, which has a separate memory space. This is not a problem on the Metal backend as it uses shared memory accessible from both the CPU and the GPU without the need to copy. The ParPy compiler could automatically insert copying when using the CUDA backend, but such copying is costly and may result in surprisingly bad performance.

## Solution

It is strongly recommended to use parallel reduction operators in place of manual for-loops where applicable. In this case, we could use `parpy.operators.sum` to implement the summation as shown below.
```python
import parpy
@parpy.jit
def sum_values(x, y):
    parpy.label('N')
    y[0] = parpy.operators.sum(x[:])
```

However, there are many situations where this does not apply. In these cases, we can use the `parpy.gpu` context manager. The ParPy compiler will ensure that all code within a `parpy.gpu` block runs on the GPU regardless of whether it contains any parallelization. Using this approach, the resulting code looks like:
```python
import parpy
@parpy.jit
def sum_values(x, y, N):
    with parpy.gpu:
        y[0] = 0.0
        parpy.label('N')
        for i in range(N):
            y[0] += x[i]
```
