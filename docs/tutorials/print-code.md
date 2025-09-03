---
sidebar_position: 3
---

# Printing generated code

In this tutorial, use the example of row-wise summation from the previous tutorial. However, instead of calling the decorated functions, we use explicit APIs exposed by ParPy to generate low-level code and later execute it. The code shown in this example is found at `examples/print.py` in the ParPy repository. From the root of the repository, run the example using `python examples/print.py`.

## Row-wise summation

In the first tutorial ([Basic Parallelization](/docs/tutorials/basic-parallelization)), we considered an annotated implementation of row-wise summation in ParPy. Below, we present the annotated version of the `row_sums` function, the allocation of test data using NumPy, and the declaration of a parallel specification.

```python
import parpy

@parpy.jit
def sum_rows(x, y, N):
    parpy.label('outer')
    for i in range(N):
        out[i] = parpy.sum(x[i,:])

import numpy as np

N = 100
M = 1024
x = np.random.randn(N, M).astype(np.float32)
y = np.empty((N,), dtype=np.float32)

p = {'outer': parpy.threads(N)}
opts = parpy.par(p)
```

For the sake of this tutorial, we assume we do not want to immediately execute the parallelized function. For instance, we may want to delay execution because we want to:

- Validate that a given parallelization strategy results in the expected low-level code.
- Test the code generation on a system where the selected target backend is unavailable.
- Be able to manually modify the generated code (e.g., to use features not accessible from Python).
- Minimize the overhead of a function call

The ParPy API exposes the `print_compiled` function for JIT-compiling a function, given a list of the arguments to be passed to the function and the compiler options. The result is a string which we can print to standard output (e.g., for debugging) or store in a file (e.g., to manually modify the generated code). For instance, to compile and print the generated code for the CUDA backend:
```python
opts.backend = parpy.CompileBackend.Cuda
code = parpy.print_compiled(sum_rows, [x, y, N], opts)
print("Generated code for CUDA:")
print(code)
print("=====")
```
or similarly for the Metal backend
```python
opts.backend = parpy.CompileBackend.Metal
code = parpy.print_compiled(sum_rows, [x, y, N], opts)
print("Generated code for Metal:")
print(code)
print("=====")
```

Assume the generated code for a function is not behaving as we expect it to. Printing it using the `print_compiled` function can help in certain cases, but it may be more helpful to manually modify the generated code (e.g., by adding custom prints inside a CUDA kernel). Given that the generated code is stored as a string in the `code` variable, we can enable modifying it at runtime by writing the code to a file and reading back after a delay:
```python
with open("out.txt", "w+") as f:
    f.write(code)

input("Press enter when finished updating 'out.txt' ")

with open("out.txt", "r") as f:
    code = f.read()
```

In this code snippet, we write the generated low-level code to a new file `out.txt`. Then, we use the `input` function in Python to wait until the user presses enter. Before pressing enter, the user modifies the generated code stored in the `out.txt` before it is read back into the `code` variable. The user could, for instance, insert debug prints in the generated code for debugging or just to help understand what the code is doing.

### Compiling and running the modified code

The ParPy compiler can generate code for any backend regardless of whether it is supported or not. However, to actually compile and run the code, a backend must be enabled (see the [installation instructions](/installation)). The following Python code compiles the modified low-level code in `code` and runs it:
```python
fn = parpy.compile_string("sum_rows", code, opts)
fn(x, y, N)
assert np.allclose(y, np.sum(x, axis=1), atol=1e-3)
```

The `compile_string` function compiles the low-level code and produces a wrapper function. We pass the name of the function in the generated code (`sum_rows`), the code to compile (`code`), and the compiler options (`opts`). The resulting wrapper function (stored in `fn`) can be used to immediately invoke the underlying JIT-compiled code. Importantly, `fn` does not expect the compiler options, as these were already provided in the call to `compile_string`.

## Specialization and API differences

As the ParPy compiler runs just-in-time (JIT), the values of all parameters passed to a function are available when it runs. The ParPy compiler will automatically specialize the generated code based on the shape of argument arrays (with a non-empty shape), and on the values of scalar parameters (e.g., floats and ints). As a result, the compiler produces more efficient code, but it may have to run many times due to variations in argument sizes.

When we call a decorated function, the wrapper code handles the specialization automatically. Based on the provided arguments and the compiler options, the wrapper code determines whether a matching specialization exists in the cache, or if another version has to be compiled. As compiling the generated code can take more than a second (depending on the target backend), using this cache can save us a lot of time. However, because the wrapper has to manage the cache, each call to a decorated function introduces a bit of overhead.

Using the `print_compiled` and `compile_string` functions results in different behavior, as the code is always specialized using `print_compiled` (i.e., there is no caching involved). As a result, the wrapper code produced by the `compile_string` function does not need to consider this caching, which reduces its overhead compared to the wrapper used in calls to decorated functions.
