---
sidebar_position: 3
---

# parpy.buffer

## Functions

### sync(backend: parpy.CompileBackend)

Synchronizes the CPU with the device of the selected `backend`. This function waits until all GPU kernels on the target device complete.

### empty(shape: Tuple[int], dtype: parpy.ElemSize, backend: parpy.CompileBackend) -> Buffer

Produces an empty `Buffer` with the specified shape `shape` containing values of data type `dtype`. The data is allocated on the backend selected using the `backend` argument.

### empty_like(b: Buffer) -> Buffer

Produces an empty `Buffer` of the same shape, data type, and backend as the provided buffer `b`.

### zeros(shape: Tuple[int], dtype: parpy.ElemSize, backend: parpy.CompileBackend) -> Buffer

Allocates a buffer in a similar vein to the `empty` function and sets all memory to zero.

### zeros_like(b: Buffer) -> Buffer

Allocates a buffer in a similar vein to the `empty_like` function and sets all memory to zero.

## Classes

### Buffer

#### sync(self)

Synchronizes the contents of this buffer with the device, by delaying execution until all GPU kernels complete.

#### from_array(t, backend) -> Buffer

Attempts to convert the provided array `t` to a Buffer of the selected backend using the [array interface protocol](https://numpy.org/doc/stable/reference/arrays.interface.html). If `backend` is set to `None`, the Buffer is constructed only for the purpose of aiding code generation and may contain an invalid pointer.

#### numpy(self) -> numpy.ndarray

Converts the buffer to a NumPy array. Depending on the buffer backend, this may result in copying data from the GPU to the CPU, and the allocation of a new buffer for the resulting NumPy array.

#### torch_ref(self) -> torch.tensor

Attempts to construct a PyTorch tensor referencing the data stored in the buffer. If the buffer backend is CUDA, this function produces a PyTorch tensor reusing the same data (this may be valuable for performance). Otherwise, it produces a new tensor containing a copy of the data in the original buffer.

#### reshape(self, \*dims) -> Buffer

Constructs a new view of the Buffer with a shape based on the provided integer dimensions `dims`. If the specified shape differs in size from the original shape, a ValueError is raised.

#### with_type(self, new_dtype) -> Buffer

Constructs a new view of the Buffer with the provided data type `dtype`. This data type must either be of the internal data type `parpy.DataType` or one of the scalar types defined in the `parpy.types` module.
