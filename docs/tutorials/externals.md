---
sidebar_position: 3
---

# External Functions

In this tutorial, we consider two use-cases of external functions to work around limitations in ParPy. For simplicity, we focus on the CUDA backend, as it has support for user-defined externals in GPU code.

The support for external functions in ParPy is limited and depends on the backend and where the external is intended to be used (CPU or GPU). Note also that the externals interface is under development, and may change drastically in the future.

## Popcount Function

Assume we want to compute the number of set bits in a 32-bit integer (often referred to as [popcount](https://en.wikipedia.org/wiki/Popcount)). In Python code, we could use the `bit_count()` method of integers to compute this efficiently (for non-negative numbers):
```python
print(123 .bit_count())
```

However, this code cannot be JIT-compiled using ParPy, as the `bit_count()` method is not supported by our compiler. Further, there is no immediate way to express this operation in ParPy. In this situation, we can declare and use an external function to work around this. We can declare a `popcount` external in ParPy for the CUDA backend using the `@parpy.external` decorator:
```python
import parpy

@parpy.external("__popc", parpy.CompileBackend.Cuda, parpy.Target.Device)
def popcount(n: parpy.types.I32) -> parpy.types.I32:
    return n.bit_count()
```

We provide three positional arguments to the `@parpy.external` decorator:
1. The name of the popcount function in CUDA (for 32-bit integers).
2. The backend where this external is available (in this case, the CUDA backend).
3. Whether the external can be called from the host (sequential code, running on the CPU) or the device (parallel code, runnning on the GPU).

Finally, the declaration uses the `parpy.types` package to declare that the function has a 32-bit integer parameter and that it produces a 32-bit integer result. By providing a function body with the expected behavior, we can call this function from sequential Python code (e.g., to verify that the generated code works as expected):
```python
print(popcount(123))
```

If we need to apply the popcount operation to a list of integers, it can be useful to perform this operation on the GPU. We would like to implement the function as:
```python
def popcount_many(x):
    return sum(popcount(x[:]))
```

Next, we allocate data for the operations:
```python
import numpy as np
N = 100
x = np.random.randint(1, 1000, (N,)).astype(np.int32)
```

Apart from the fact that this function does not include the necessary labels, there are three main issues that prevent us from implementing the function in this way in ParPy:
1. Currently, ParPy does not support returning a value from a JIT-compiled function. To work around this, we have to provide a singleton array as an argument, and have the function write the result there.
2. The ParPy compiler has limited support for slicing operations, and they cannot be used within function calls. Instead, we can implement the summation using a for-loop.
3. The label associated with the summation for-loop must specify that we want to perform a parallel reduction, as the ParPy compiler does not infer this manually.

Taking these three points into account, we can implement the `popcount_many` function in ParPy as:
```python
@parpy.jit
def popcount_many(x, count, N):
    parpy.label('N')
    for i in range(N):
        count[0] += popcount(x[i])
```
and we can run it as follows, assuming the values `x` and `N` are initialized as shown above:
```
p = {'N': parpy.threads(N).reduce()}
opts = parpy.par(p)
count = np.array([0], dtype=np.int32)
popcount_many(x, count, N, opts=opts)
```

A complete version of this example can be found at `examples/external.py` in the ParPy repository.

## Custom Warp Summation

To achieve peak performance, an advanced user may sometimes need to write manual code. The previous example showed how we can use existing functions from the standard library. However, what if we want to perform an operation that involves multiple threads, such as a parallel reduction? In this example, we implement a custom warp summation in CUDA, which uses different intrinsics depending on the capabilities of the target GPU (while ParPy currently uses the most general alternative). Below, we present the CUDA C++ implementation of such a function. For simplicity, this function assumes we are summing exactly 32 values (the size of a warp).
```cpp
__device__ int32_t warp_sum(int32_t *values) {
  int32_t v = values[threadIdx.x];
#if (__CUDA_ARCH__ >= 800)
  return __reduce_add_sync(0xFFFFFFFF, v);
#else
  for (int i = 16; i > 0; i /= 2) {
    v = __shfl_xor_sync(0xFFFFFFFF, v, i);
  }
  return v;
#endf
}
```

The former alternative uses the `__reduce_add_sync` intrinsic that is available only on GPUs with compute capability 8.x or higher, while the latter uses the same approach as in ParPy. This function expects to run on exactly 32 threads, and we assume it is stored in a header file named `cuda_helper.h`. We specify this in the external declaration using optional keyword arguments:
```python
import parpy
@parpy.external(
    "warp_sum", parpy.CompileBackend.Cuda, parpy.Target.Device,
    header="<cuda_helper.h>", parallelize=parpy.threads(32)
)
def warp_sum(x: parpy.types.pointer(parpy.types.I32)) -> parpy.types.I32:
    return sum(x)
```

The function declaration specifies that the argument type is a pointer to 32-bit integers and that the result is a 32-bit integer. It also provides a default implementation to be able to run the function from Python.

Next, we define a row-wise summation based on this external function:
```python
@parpy.jit
def sum_rows(x, y, N):
    parpy.label('N')
    for i in range(N):
        y[i] = warp_sum(x[i])
```

Finally, we have to add the path to the header file to the include path. To do this, we add it to the `includes` field of the compiler options before calling the function:
```python
import os
p = {'N': parpy.threads(N)}
opts = parpy.par(p)
include_path = f"{os.path.dirname(os.path.realpath(__file__))}/code"
opts.includes += [include_path]
sum_rows(x, y, N, opts=opts)
assert np.allclose(y, np.sum(x, axis=1))
```

This example is available in `examples/ext-sum.py` and the CUDA C++ code is found at `examples/code/cuda_helper.h`.
