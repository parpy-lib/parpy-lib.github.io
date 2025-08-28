# Inconsistent parallel structure

Consider the following code normalizing the rows of an input matrix `x` and storing the result in an output matrix `y`.
```python
import numpy as np
import parpy

@parpy.jit
def normalize_rows(x, y, N):
    parpy.label('N')
    for i in range(N):
        parpy.label('M1')
        s = parpy.operators.sum(x[i, :])

        parpy.label('M2')
        y[i, :] = x[i, :] / s

N = 100
M = 250
x = np.random.randn(N, M).astype(np.float32)
y = np.empty_like(x)
opts = parpy.par({
    'N': parpy.threads(N),
    'M1': parpy.threads(128),
    'M2': parpy.threads(256),
})
opts.backend = parpy.CompileBackend.Metal
print(parpy.print_compiled(normalize_rows, [x, y, N], opts))
```

When we run this code, the ParPy compiler produces an error indicating that some parallel structures are inconsistent.

## Problem

The issue with this code is the parallel specification provided to the `parpy.par` function. Importantly, it specifies that the summation (labeled `M1`) should run on `128` threads, while the elementwise division (labeled `M2`) should run on `256` threads. This parallelism is inconsistent, which is problematic because we have to decide how many threads to run when we launch the generated GPU kernel. Allowing either operation to run with fewer threads would be suboptimal, as the remaining threads would be idle during that time. Therefore, there are two alternative ways in which the compiler can resolve this.

The first alternative is for the compiler to force either operation to use the same number of threads as the other. However, this would violate the user's parallel specification. In particular, if the compiler was to force the use of `128` threads, this may result in less parallelism than expected, thereby reducing performance.

The second option is for the compiler to automatically break up the two loops into separate loop nests. However, the resulting code in this case will launch two kernels instead of one, leading to an increased overhead. In this case, this has a negligible cost, but if these kernels were launched many times, this can have a noticeable impact on performance. For this reason, we argue that this is also a poor solution. Instead, the user has to manually resolve this.

## Solution

To resolve this, the user has to manually implement either of the two alternatives proposed above. One option is to adjust the parallel specification such that both inner parallel operations use the same number of threads. For example, we could resolve this by adjusting the specification as follows:
```python
opts = parpy.par({
    'N': parpy.threads(N),
    'M1': parpy.threads(128),
    'M2': parpy.threads(256),
})
```

Alternatively, if the provided parallelism is considered important, the user can resolve this by placing the code in two separate parallel loop nests. This way, the resulting code ends up as two separate kernels, each of which have a consistent parallel structure. To achieve this, we could update the implementation of `normalize_rows` as follows:
```python
@parpy.jit
def normalize_rows(x, y, N):
    parpy.label('N')
    for i in range(N):
        parpy.label('M1')
        s = parpy.operators.sum(x[i, :])

    parpy.label('N')
    for i in range(N):
        parpy.label('M2')
        y[i, :] = x[i, :] / s
```
