---
sidebar_position: 1
---

# parpy

This module exposes the key functionality of ParPy needed to parallelize functions.

## Functions

### jit(Function) -> Function

The `jit` function is typically used as a decorator on a Python function to be just-in-time (JIT) compiled. For example,
```python
import parpy
@parpy.jit
def sum_values(x, y):
    ...
```
decorates the function `sum_values`. As a result, the decorated function is parsed and translated to an intermediate representation, such that a wrapper function is returned in place of the original Python function. When we call the `sum_values` function, this wrapper function runs instead. It performs a JIT-compilation based on the shapes and types of the provided arguments before executing the generated native code.

Note that this wrapper function performs caching to avoid having to repeatedly JIT-compile a function.

### external(ext_name: string, backend: CompileBackend, target: Target, header: string, parallelize: LoopPar)(Function) -> Function

The `external` decorator is used on any Python function to make its name and types represent an external function accessible from a JIT-compiled ParPy function. This decorator takes five arguments, and produces a function that takes the decorated function as its sole argument. A function decorated with `external` can be used as a regular Python function afterward, but its details are recorded internally by the ParPy library.

The `ext_name` argument specifies the name of the external function. This name may be equal to the name of the Python function, but it does not have to be equal. If a JIT-compiled ParPy function refers to this external, it will use the string provided in `ext_name` to refer to the function in the generated low-level code. The `backend` argument determines which backend the external is defined for, and the `target` argument specifies whether the external runs on the host (CPU) or the device (GPU).

The optional `header` argument can be used to specify the name of a header in which the external function is defined. This is required if the external is defined by the user or in a header that is not implicitly included. For instance, if the external refers to an intrinsic function in CUDA or Metal, it typically does not need to be included from a header. When a header is specified, the user must add the directory of the header to the include path using the `includes` field of the compiler options.

The optional `parallelize` argument can be used when the external function is intended to execute in parallel. This may only be set when the `target` is set to `Target.Device`. When a parallelization is provided via this argument, the external function will be treated as a parallel for-loop with the specified `LoopPar`. As a result, the generated code will call the external function with the same arguments across several threads. The user is responsible for ensuring all threads agree on the returned value from the external function.

The following constraints apply to the combination of backend and target:
- CUDA:
    - Host: Not supported. As all data is allocated on the GPU, such an external would require implicit copying of data between the CPU and GPU which may be costly.
    - Device: Supported. The user must ensure external functions are annotated with `__device__` to avoid errors from the CUDA compiler.
- Metal:
    - Host: Supported.
    - Device: Supported for functions available in the Metal standard library. The ParPy compiler uses an API for compiling Metal that does not allow overriding the include path. Therefore, user-defined functions are not supported.

### print_compiled(fun: Function, args: List[Any], opts: CompileOptions) -> string

This function takes three arguments: a function to compile (`fun`), a list of arguments (`args`), and the compiler options (`opts)`. The function to be compiled can be any Python function (i.e., it does not have to be decorated to prepare for JIT-compilation), but it must follow the restrictions of the ParPy compiler. The function attempts to compile the provided function and returns the low-level code as a string, when compilation succeeds. Otherwise, when it fails (e.g., because the function uses unsupported features), it will raise an error containing details about the error.

The provided arguments are used for specialization. A scalar argument is inlined into the generated code, while the dimensions of an array argument are hard-coded into the generated code (but not its contents). Importantly, this means that the generated function can be reused given that:
- We provide the same scalar arguments, and
- All array arguments have the same shape (but not necessarily the same contents)

If the compiler options are not provided, the function will use the default set of compiler options.

### compile_string(fun_name: string, code: string, opts: CompileOptions) -> Function

This function takes three arguments: the name of the native function to invoke (`fun_name`), the generated low-level code (`code`), and the compiler options (`opts`). The function compiles the code to a shared library and returns a wrapper function which validates the provided arguments before calling the native function in the shared library.

If the compiler options are not provided, the function will use the default set of compiler options.

### threads(n: int) -> LoopPar

Returns a `LoopPar` object requesting the specified number of threads.

### reduce() -> LoopPar

Returns a `LoopPar` object with its `reduce` field set to `True`.

### par(p: Dict[string, LoopPar]) -> CompileOptions

Returns the default `CompileOptions` object with parallel execution enabled based on the provided parallel specification `p`.

### clear_cache

Clears the cached shared library files as well as the runtime cache of functions.

## Operators

These operators are exposed in the main package because they are very useful when defining a ParPy function.

### gpu

This operator is used to force sequential code to execute on the GPU. By default, the compiler will place sequential code on the CPU side, but using this [context manager](https://docs.python.org/3/library/contextlib.html) you can force the compiler to run all code placed within the context on the GPU. In the Python interpreter, this operator is a no-op.

### label

This operator is used for labeling statements in the code. Labels are referred to when 

## Types

### CompileOptions

This structure contains options used to control different aspects of the compiler. Each option is documented by specifying its type, the default value, and how it impacts the code generation.

#### parallelize : Dict[string, LoopPar]

Defines the parallel specification, which defines a mapping from labels to their parallelization. This is automatically set when using the `seq` and `par` functions. By default, it is set to an empty dictionary.

#### verbose_backend_resolution : bool

When enabled, the compiler will print detailed information on why backends are considered disabled. By default, this flag is set to `False`.

#### debug_print : bool

When enabled, the native compiler will print a number of intermediate AST representations at various points in the compilation process, along with timing information (relative to the start of the native compilation). Can be useful when debugging issues in the compiler. By default, this flag is set to `False`.

#### write_output : bool

If the compilation of the generated low-level code fails, and this flag is enabled, the compiler will store the low-level code in a temporary file and include its path in the error message. By default, this flag is set to `False`.

#### backend : parpy.CompileBackend

Determines which backend to use for code generation. By default, this is set to `CompileBackend.Auto`.

#### use_cuda_thread_block_clusters : bool

When using the CUDA backend and this flag is enabled, the compiler will make use of thread block clusters instead of relying on the inter-block synchronization for parallel reductions. This applies only when the number of threads are less than the maximum number of threads per blocks times the maximum number of thread blocks used per cluster (set in another option). By default, this option is set to `False`.

Note that the ParPy compiler will not automatically determine whether to enable this option or not based on the target architecture. This is because it may not always be beneficial, and also, it may be interesting to enable to see what the generated code looks like even on a machine that lacks support for it.

#### max_thread_blocks_per_cluster : int

When the use of thread block clusters is enabled, this option can be used to override the maximum number of thread blocks per cluster. Currently, as of CUDA 13.0, the maximum number of thread blocks per cluster guaranteed to be supported is `8` (which is the default value of this option). However, certain GPUs may support a larger number of thread blocks in a cluster (e.g., an H100 GPU supports up to `16`).

Note that this value must be set to a power of two.

#### use_cuda_graphs : bool

When using the CUDA backend and this flag is enabled, the compiler emits code utilizing [CUDA graphs](https://developer.nvidia.com/blog/cuda-graphs/). When the generated CUDA C++ code involves a large number of GPU kernel launches, using CUDA graphs can drastically improve performance by informing CUDA about all intended kernel launches up-front. However, as this may have a negative performance impact, and will not provide any meaningful benefits when code involves only a few kernels (as is often the case), this option is set to `False` by default.

#### force_int_size : Option[types.ElemSize]

When set to a value, the native compiler uses the specified type for all integer values throughout the compiler (except where the user overrides this via conversion operators). By default, this is set to `None`.

When this option is `None`, the compiler automatically determines the size to use based on the selected backend:
- In the CUDA backend, integers are 64-bit by default
- In the Metal backend, integers are 32-bit by default

#### force_float_size : Option[types.ElemSize]

When set to a value, the native compiler uses the specified type for all floating-point values, in the same manner as the `force_int_size` option. By default, this is set to `None`. Note that the Metal backend does not support 64-bit floating-point numbers, but the ParPy compiler will not prevent users from forcing the use of 64-bit floats on the Metal backend.

When this option is `None`, the compiler automatically determines the size to use based on the selected backend:
- In the CUDA backend, floats are 64-bit by default
- In the Metal backend, floats are 32-bit by default

#### includes : List[string]

Contains a list of directories to be added to the include path, used when compiling the generated low-level code. By default, this is an empty list.

#### libs : List[string]

Contains a list of directories to be added to the library path, used when compiling the generated low-level code. By default, this is an empty list.

#### extra_flags : List[string]

Contains a list of additional flags to be provided to the underlying compiler used to produce a shared library from the generated low-level code. By default, this is an empty list.

The underlying compiler is `nvcc` for the CUDA backend and `clang++` for the Metal backend.

### CompileBackend

This is an enum type representing a backend. Currently, it supports the following values:
- `CompileBackend.Auto`
- `CompileBackend.Cuda`
- `CompileBackend.Metal`

The `Auto` backend is not an actually supported backend, but it used to indicate that the compiler should automatically select the target backend.

### LoopPar

A type used to represent the parallelization of a statement (typically, a loop). A default value of this type can be constructed using its constructor (which takes no arguments). It can only be manipulated via methods that produce a new version with an overriden property.

#### threads(self, nthreads: int) -> LoopPar

Returns a new `LoopPar` with the specified number of threads (`nthreads`).

#### reduce(self) -> LoopPar

Returns a new `LoopPar` where the `reduce` field is set to `True`. This is not required when using the reduction operators. However, the ParPy compiler does not automatically identify for-loops that perform a reduction. For instance, to correctly parallelize the below for-loop, we have to map the label `N` to a `LoopPar` on which the `reduce` field is set to `True`.
```python
parpy.label('N')
for i in range(N):
    x[0] += y[i]
```

#### tpb(self, tpb: int) -> LoopPar

Returns a new `LoopPar` enforcing that the resulting code uses the specified number of threads per block. By default, the compiler uses `1024` threads per block. When overridden to any other value, this is accepted instead.

Note that using `LoopPar`s with conflicting (non-default) numbers of threads per block in a single parallel loop nest will result in an compiler error.

### ElemSize

Represents the element size of an array in ParPy. The following values are supported by the ParPy compiler:
- `Bool`
- `I8`
- `I16`
- `I32`
- `I64`
- `U8`
- `U16`
- `U32`
- `U64`
- `F16`
- `F32`
- `F64`

These types correspond to booleans (`Bool`), signed integers (`I8`, `I16`, `I32`, and `I64`), unsigned integers (`U8`, `U16`, `U32`, and `U64), and floating-point numbers (`F16`, `F32`, and `F64`).

### Target

An enum used in the declaration of an external function, to indicate the intended target at which the external should run. Contains two values: `Target.Host` and `Target.Device`.

Using `Target.Host` indicates that the external function is available in host code (i.e., on the CPU). As such, this function must not be called from parallel code. On the other hand, the `Target.Device` indicates that the external function is defined in device code (i.e., on the GPU). This function may only be called from parallel code.

Note that when using the CUDA backend, an external function set to use the `Target.Device` target must be defined using the `__device__` attribute.
