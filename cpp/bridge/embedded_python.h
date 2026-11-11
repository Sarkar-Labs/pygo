#pragma once

// Embedded CPython host interface used by the Go-first runtime.  The Go
// process owns the interpreter; this layer only provides native registration
// and lifecycle helpers.  No subprocess or Python sidecar is involved.

#ifdef __cplusplus
extern "C" {
#endif

// Register PyGo's statically linked pybind11 extension modules with CPython.
// Must be called before Py_Initialize(). Returns 0 on success, -1 on error.
int pygo_python_register_modules(void);

// Return whether the native pybind11 extension was compiled into libpygo.
int pygo_python_has_native_module(void);

#ifdef __cplusplus
}
#endif
