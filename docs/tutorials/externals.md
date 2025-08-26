---
sidebar_position: 3
---

# External Functions

In this tutorial, we consider two use-cases of external functions to work around limitations in ParPy.

## Popcount Function

Assume we want to compute the number of set bits in a 32-bit integer (often referred to as [popcount](https://en.wikipedia.org/wiki/Popcount)). In Python code, we could use the `bit_count()` method of integers to compute this efficiently (for non-negative numbers):
```python
print(123.bit_count())
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
