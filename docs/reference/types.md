---
sidebar_position: 4
---

# parpy.types

This module provides attributes and functions used to represent types. These can be used when declaring the types of arguments and return values for external functions, and when specifying the type of elements stored in a buffer.

## Functions

### pointer(ty: parpy.ElemSize) -> parpy.ExtType

Constructs a pointer type based on the given type `ty`, which must be an `ElemSize` type. This produces a distinct type to prevent the construction of nested pointer types.

## Type Attributes

The module provides a pre-defined list of external type attributes based on the `ElemSize` type.
