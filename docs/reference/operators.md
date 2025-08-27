---
sidebar_position: 5
---

# parpy.operators

This module defines operators that are used in JIT-compiled code. When used in Python code, the operators either rely on existing implementations (the math operations) or perform a no-op (the conversion functions).

## Reduction Operators

Currently, this module includes the following parallelizable reduction operators:
- `max`
- `min`
- `prod`
- `sum`

These operators expect an array of type `Array[T]` as input and produce a scalar value of type `T`. When parallelized, the ParPy compiler will generate optimized code for performing the parallel reductions, by making use of intrinsics of the selected backend to efficiently share data among threads.

## Arithmetic Operators

This module defines the following arithmetic operators, which are supported by the ParPy compiler:
- `abs`
- `atan2`
- `cos`
- `exp`
- `log`
- `max` (as a binary operator)
- `min` (as a binary operator)
- `sin`
- `sqrt`
- `tanh`

When used outside of JIT-compiled code, these functions rely either on NumPy or built-in operators.

## Conversion Operators

This module also provides a definition of operators used to force conversion of a scalar value to a particular type. These are useful when it is important that a scalar has a particular size (for instance, when using bitwise operations). The following conversion functions are provided:
- `float16`
- `float32`
- `float64`
- `int8`
- `int16`
- `int32`
- `int64`
- `uint8`
- `uint16`
- `uint32`
- `uint64`

## Attributes

### inf

Representation of the floating-point number for infinity. If the compiler options sets a value for the `force_float` option, this size is used to represent the infinity. Otherwise, it is determined based on the target backend.
