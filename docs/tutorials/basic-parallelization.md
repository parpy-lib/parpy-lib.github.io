---
sidebar_position: 1
---

# Basic Parallelization

In this tutorial, we consider a simple Python function and how we can parallelize it to run in parallel on the GPU. The code shown in this example is found at `examples/hello.py` in the ParPy repository.

## Row-Wise Summation

Assume we have an implementation of row-wise summation in Python that we want to parallelize:
```python
def sum_rows(x, out, N):
  for i in range(N):
      out[i] = sum(x[i,:])
```

To be able to parallelize this function using ParPy, we have to annotate it. First, we add the function decorator `@parpy.jit` on a line before the function definition. This enables the function to be called with customizable parallelization. Second, we add labels using `parpy.label` associated with the subsequent statement. These labels are used when we specify how to parallelize the function. In this case, we only associate a label with the outer for-loop of the function. The resulting annotated function looks like:

```python
import parpy

@parpy.jit
def sum_rows(x, y, N):
    parpy.label('outer')
    for i in range(N):
        out[i] = parpy.operators.sum(x[i,:])
```

We allocate randomized problem data using NumPy:
```python
import numpy as np

N = 100
M = 1024
x = np.random.randn(N, M).astype(np.float32)
y = np.empty((N,), dtype=np.float32)
```

When we call the function, we determine how to parallelize it by referring to the labels declared in the function body. To have the function run in parallel over each row of the matrix `x` (i.e., by parallelizing the outer loop), we define a parallel specification `p`, and construct a compilation options object `opts` based on this, as follows
```python
p = {'outer': parpy.threads(N)}
opts = parpy.par(p)
```

Assuming either the CUDA or Metal backends have been properly set up, we can now run the function with parallelization by providing the `opts` keyword argument:
```python
sum_rows(x, y, N, opts=opts)
assert np.allclose(y, np.sum(x, axis=1), atol=1e-3)
```

The compiler options `opts` include a parallel specification, but also includes other options related to the code generation, such as which backend to use. By default, the compiler automatically determines which backend to use.
