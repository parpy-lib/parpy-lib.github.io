---
sidebar_position: 2
---

# parpy.backend

## Functions

### is_enabled(backend: parpy.CompileBackend) -> bool

Determines whether the provided backend `backend` is enabled. A backend must be enabled to be able to use it as a target when JIT-compiling functions.

## Attributes

### available : List[parpy.CompileBackend]

Contains a list of the available (enabled) backends of ParPy.

### backends : List[parpy.CompileBackend]

Contains a list of all backends for which ParPy supports codegen.
