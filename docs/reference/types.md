---
sidebar_position: 4
---

# parpy.types

This module provides attributes and functions used to represent types. These can be used when declaring the types of arguments and return values for external functions, and when specifying the type of elements stored in a buffer.

## Functions

### pointer(ty: parpy.ExtType) -> ExtType

Constructs a pointer type based on the given type `ty`, which must be a scalar type (nested pointers are not allowed).

## Types

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

## Type Attributes

The module provides a pre-defined list of external type attributes of type `ExtType`, named according to the values provided in the `ElemSize` type.
