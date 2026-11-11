#include "embedded_python.h"

#include <Python.h>

namespace {
#if defined(PYGO_PYBIND_ENABLED)
extern "C" PyObject* PyInit__pygo_native();
#endif
}

extern "C" int pygo_python_register_modules(void) {
#if defined(PYGO_PYBIND_ENABLED)
    // PyImport_AppendInittab must run before interpreter initialization.
    if (PyImport_AppendInittab("_pygo_native", &PyInit__pygo_native) == -1) {
        return -1;
    }
    return 0;
#else
    return 0;
#endif
}

extern "C" int pygo_python_has_native_module(void) {
#if defined(PYGO_PYBIND_ENABLED)
    return 1;
#else
    return 0;
#endif
}
