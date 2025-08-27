---
sidebar_position: 3
---

# parpy.buffer

## Functions

### sync(backend: parpy.CompileBackend)

Synchronizes the CPU with the device of the selected `backend`. This function waits until all GPU kernels on the target device complete.

## Classes

### Buffer

#### sync(self)

Synchronizes the contents of this buffer with the device, by delaying execution until all GPU kernels complete.

#### from_array(t, backend) -> Buffer

Attempts to convert the provided array `t` to a Buffer of the selected backend using the [array interface protocol](https://numpy.org/doc/stable/reference/arrays.interface.html). If `backend` is set to `None`, the Buffer is constructed only for the purpose of aiding code generation and may contain an invalid pointer.

#### numpy(self) -> numpy.ndarray

Converts the buffer to a NumPy array. Depending on the buffer backend, this may result in copying data from the GPU to the CPU, and the allocation of a new buffer for the resulting NumPy array.

#### reshape(self, \*dims) -> Buffer

Constructs a new view of the Buffer with a shape based on the provided integer dimensions `dims`. If the specified shape differs in size from the original shape, a ValueError is raised.

#### with_type(self, new_dtype) -> Buffer

Constructs a new view of the Buffer with the provided data type `dtype`. This data type must either be of the internal data type `parpy.DataType` or one of the scalar types defined in the `parpy.types` module.
