// Package cgo provides the CGo bridge between Go and Python/C++ layers.
// This is the most critical file in PyGo — it enables zero-copy tensor
// sharing between Go, Python (CPython), and C++ compute kernels.
package cgo

/*
#cgo pkg-config: python3-embed
#cgo LDFLAGS: -lm -ldl
#cgo CFLAGS: -I../../cpp/tensor


#include <Python.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "../../cpp/bridge/embedded_python.h"
typedef struct pygo_tensor_handle pygo_tensor_handle;
#if defined(__GNUC__) || defined(__clang__)
extern pygo_tensor_handle* pygo_tensor_zeros(const int64_t*, size_t) __attribute__((weak));
extern void pygo_tensor_release(pygo_tensor_handle*) __attribute__((weak));
extern float* pygo_tensor_mutable_data_f32(pygo_tensor_handle*) __attribute__((weak));
#else
extern pygo_tensor_handle* pygo_tensor_zeros(const int64_t*, size_t);
extern void pygo_tensor_release(pygo_tensor_handle*);
extern float* pygo_tensor_mutable_data_f32(pygo_tensor_handle*);
#endif
#if defined(__GNUC__) || defined(__clang__)
extern int pygo_python_register_modules(void) __attribute__((weak));
extern int pygo_python_has_native_module(void) __attribute__((weak));
#endif

// ─── Python Runtime ───────────────────────────────────────────────────────────

static int pygo_init_python(const char* program) {
    if (Py_IsInitialized()) return 0;
#if defined(__GNUC__) || defined(__clang__)
    if (pygo_python_register_modules && pygo_python_register_modules() != 0) return -1;
#else
    if (pygo_python_register_modules() != 0) return -1;
#endif
    PyConfig config;
    PyConfig_InitPythonConfig(&config);
    config.isolated = 0;
    config.use_environment = 1;
    config.site_import = 1;
    if (program && *program) {
        PyStatus program_status = PyConfig_SetBytesString(&config, &config.program_name, program);
        if (PyStatus_Exception(program_status)) {
            PyConfig_Clear(&config);
            return -1;
        }
    }
    PyStatus status = Py_InitializeFromConfig(&config);
    PyConfig_Clear(&config);
    if (PyStatus_Exception(status)) {
        if (PyStatus_IsExit(status)) return (int)status.exitcode;
        return -1;
    }
    return Py_IsInitialized() ? 0 : -1;
}

static void pygo_finalize_python() {
    if (Py_IsInitialized()) Py_Finalize();
}

static int pygo_is_initialized() {
    return Py_IsInitialized();
}

// ─── Tensor Bridge ────────────────────────────────────────────────────────────

typedef struct {
    void*    data_ptr;    // Shared native allocation visible to Go/Python/C++
    int64_t* shape;       // Dimension sizes
    int64_t* strides;     // Strides in bytes
    int      ndim;        // Number of dimensions
    int      dtype;       // 0=f32, 1=f16, 2=bf16, 3=f64, 4=i8, 5=i64
    int      device;      // 0=cpu, 1=cuda:0, 2=cuda:1, ...
    int64_t  numel;       // Total number of elements
    int      itemsize;    // Bytes per element
    int      owns_data;   // Native allocation ownership flag
    size_t   refs;        // Go + Python capsule references
    pygo_tensor_handle* native_handle; // C++ Tensor engine handle when available
} PyGoTensor;

static PyGoTensor* pygo_tensor_create(
    void* data, int64_t* shape, int64_t* strides,
    int ndim, int dtype, int device, int64_t numel, int itemsize
) {
    if (ndim < 0 || (ndim > 0 && (!shape || !strides))) return NULL;
    PyGoTensor* t = (PyGoTensor*)calloc(1, sizeof(PyGoTensor));
    if (!t) return NULL;
    size_t nbytes = numel > 0 ? (size_t)numel * (size_t)itemsize : 0;
    if (!data && nbytes) {
        t->data_ptr = malloc(nbytes);
        t->owns_data = 1;
        if (!t->data_ptr) { free(t); return NULL; }
    } else {
        t->data_ptr = data;
        t->owns_data = 0;
    }
    t->shape = ndim ? (int64_t*)malloc((size_t)ndim * sizeof(int64_t)) : NULL;
    t->strides = ndim ? (int64_t*)malloc((size_t)ndim * sizeof(int64_t)) : NULL;
    if ((ndim && !t->shape) || (ndim && !t->strides)) {
        if (t->owns_data) free(t->data_ptr);
        free(t->shape); free(t->strides); free(t); return NULL;
    }
    if (ndim) {
        memcpy(t->shape, shape, (size_t)ndim * sizeof(int64_t));
        memcpy(t->strides, strides, (size_t)ndim * sizeof(int64_t));
    }
    t->ndim = ndim; t->dtype = dtype; t->device = device;
    t->numel = numel; t->itemsize = itemsize; t->refs = 1;
    t->native_handle = NULL;
    // Bridge-created float32 CPU tensors are backed by the real C++ Tensor
    // engine. The bridge metadata and the C++ Tensor point at the same
    // allocation, so Go/Python/C++ share storage without a copy.
    if (!data && dtype == 0 && device == 0 && pygo_tensor_zeros && pygo_tensor_mutable_data_f32 && pygo_tensor_release) {
        t->native_handle = pygo_tensor_zeros(shape, (size_t)ndim);
        if (!t->native_handle) { free(t->shape); free(t->strides); free(t); return NULL; }
        t->data_ptr = pygo_tensor_mutable_data_f32(t->native_handle);
        if (numel > 0 && !t->data_ptr) {
            if (pygo_tensor_release) pygo_tensor_release(t->native_handle); free(t->shape); free(t->strides); free(t); return NULL;
        }
    }
    return t;
}

static void pygo_tensor_retain_internal(PyGoTensor* t) {
    if (t) __atomic_add_fetch(&t->refs, 1, __ATOMIC_RELAXED);
}

static void pygo_tensor_free(PyGoTensor* t) {
    if (!t) return;
    if (__atomic_sub_fetch(&t->refs, 1, __ATOMIC_ACQ_REL) != 0) return;
    if (t->native_handle && pygo_tensor_release) pygo_tensor_release(t->native_handle);
    if (t->owns_data) free(t->data_ptr);
    free(t->shape); free(t->strides); free(t);
}

static void* pygo_tensor_data(PyGoTensor* t) {
    return t ? t->data_ptr : NULL;
}

static int64_t pygo_tensor_shape(PyGoTensor* t, int dim) {
    if (!t || dim < 0 || dim >= t->ndim) return -1;
    return t->shape[dim];
}

// ─── Python Module Loader ─────────────────────────────────────────────────────

static PyObject* pygo_import_module(const char* name) {
    return PyImport_ImportModule(name);
}

static PyObject* pygo_call_function(
    PyObject* module, const char* func_name,
    PyObject* args, PyObject* kwargs
) {
    if (!module || !func_name) return NULL;
    PyObject* fn = PyObject_GetAttrString(module, func_name);
    if (!fn || !PyCallable_Check(fn)) {
        PyErr_Format(PyExc_AttributeError, "Module has no callable '%s'", func_name);
        Py_XDECREF(fn);
        return NULL;
    }
    PyObject* owned_args = NULL;
    PyObject* call_args = args;
    if (!call_args) {
        owned_args = PyTuple_New(0);
        if (!owned_args) { Py_DECREF(fn); return NULL; }
        call_args = owned_args;
    }
    PyObject* result = PyObject_Call(fn, call_args, kwargs);
    Py_XDECREF(owned_args);
    Py_DECREF(fn);
    return result;
}

static int pygo_run_string(const char* code) {
    if (!code) { PyErr_SetString(PyExc_TypeError, "code is null"); return -1; }
    PyObject* main_mod = PyImport_AddModule("__main__");
    if (!main_mod) return -1;
    PyObject* globals = PyModule_GetDict(main_mod);
    PyObject* result = PyRun_StringFlags(code, Py_file_input, globals, globals, NULL);
    if (!result) return -1;
    Py_DECREF(result);
    return 0;
}

static int pygo_get_error(char* out, size_t out_size) {
    if (!out || out_size == 0) return 0;
    out[0] = '\0';
    if (!PyErr_Occurred()) return 0;
    PyObject *type = NULL, *value = NULL, *traceback = NULL;
    PyErr_Fetch(&type, &value, &traceback);
    PyErr_NormalizeException(&type, &value, &traceback);
    PyObject* str = value ? PyObject_Str(value) : NULL;
    const char* msg = str ? PyUnicode_AsUTF8(str) : NULL;
    if (!msg) msg = "unknown Python error";
    strncpy(out, msg, out_size - 1);
    out[out_size - 1] = '\0';
    Py_XDECREF(type); Py_XDECREF(value); Py_XDECREF(traceback); Py_XDECREF(str);
    return 1;
}

// ─── Tensor → Python capsule (zero-copy) ─────────────────────────────────────

static void _tensor_capsule_destructor(PyObject* cap) {
    if (!cap || !PyCapsule_IsValid(cap, "PyGoTensor")) return;
    PyGoTensor* t = (PyGoTensor*)PyCapsule_GetPointer(cap, "PyGoTensor");
    pygo_tensor_free(t);
}

static PyObject* pygo_tensor_to_python(PyGoTensor* t) {
    if (!t) { Py_RETURN_NONE; }
    pygo_tensor_retain_internal(t);
    PyObject* cap = PyCapsule_New(t, "PyGoTensor", _tensor_capsule_destructor);
    if (!cap) pygo_tensor_free(t);
    return cap;
}

static PyGoTensor* pygo_tensor_from_python(PyObject* cap) {
    if (!cap || !PyCapsule_IsValid(cap, "PyGoTensor")) return NULL;
    return (PyGoTensor*)PyCapsule_GetPointer(cap, "PyGoTensor");
}

// ─── GIL Management ───────────────────────────────────────────────────────────

static PyGILState_STATE pygo_gil_acquire() {
    return PyGILState_Ensure();
}

static void pygo_gil_release(PyGILState_STATE state) {
    PyGILState_Release(state);
}

static int pygo_has_native_module(void) {
#if defined(__GNUC__) || defined(__clang__)
    return pygo_python_has_native_module ? pygo_python_has_native_module() : 0;
#else
    return pygo_python_has_native_module();
#endif
}
*/
import "C"

import (
	"fmt"
	"runtime"
	"sync"
	"unsafe"
)

// ─── Python Runtime ───────────────────────────────────────────────────────────

var (
	runtimeMu sync.Mutex
	initErr   error
	finalized bool
)

// InitPython initializes the embedded CPython interpreter. It is safe to call
// repeatedly while the interpreter is alive. Once FinalizePython is called,
// reinitialization is rejected because embedded extension modules can retain
// interpreter-global state across finalization.
func InitPython() error {
	runtimeMu.Lock()
	defer runtimeMu.Unlock()
	if C.pygo_is_initialized() != 0 {
		return nil
	}
	if finalized {
		return fmt.Errorf("CPython runtime has already been finalized and cannot be reinitialized safely")
	}
	prog := C.CString(runtime.GOROOT())
	defer C.free(unsafe.Pointer(prog))
	if C.pygo_init_python(prog) != 0 {
		initErr = fmt.Errorf("failed to initialize CPython")
		return initErr
	}
	initErr = nil
	return nil
}

// FinalizePython permanently shuts down the embedded CPython interpreter for
// this process.
func FinalizePython() {
	runtimeMu.Lock()
	defer runtimeMu.Unlock()
	if C.pygo_is_initialized() != 0 {
		C.pygo_finalize_python()
	}
	finalized = true
}

// IsInitialized returns true if CPython is running.
func IsInitialized() bool {
	return C.pygo_is_initialized() != 0
}

// ─── GIL Guard ────────────────────────────────────────────────────────────────

// GIL manages the Python Global Interpreter Lock.
type GIL struct{ state C.PyGILState_STATE }

// AcquireGIL acquires the Python GIL. Caller must call Release().
func AcquireGIL() *GIL { return &GIL{state: C.pygo_gil_acquire()} }

// Release releases the GIL.
func (g *GIL) Release() { C.pygo_gil_release(g.state) }

// WithGIL runs fn while holding the GIL.
func WithGIL(fn func()) {
	gil := AcquireGIL()
	defer gil.Release()
	fn()
}

// ─── Tensor Bridge ────────────────────────────────────────────────────────────

// DType matches tensor/tensor.go DType constants.
type DType int

const (
	Float32 DType = iota
	Float16
	BFloat16
	Float64
	Int8
	Int64
)

func (d DType) ItemSize() int {
	switch d {
	case Float32:
		return 4
	case Float16, BFloat16:
		return 2
	case Float64:
		return 8
	case Int8:
		return 1
	case Int64:
		return 8
	default:
		return 4
	}
}

// BridgeTensor is a zero-copy tensor handle shared between Go, Python, and C++.
// Tensor bytes live in a native allocation so the CPython capsule can safely
// retain the storage after the Go wrapper becomes unreachable.
type BridgeTensor struct {
	ptr    *C.PyGoTensor
	shape  []int64
	dtype  DType
	device int
}

// NewBridgeTensor creates a tensor backed by a native allocation shared by Go and Python.
func NewBridgeTensor(shape []int64, dtype DType, device int) (*BridgeTensor, error) {
	numel := int64(1)
	for _, d := range shape {
		if d < 0 {
			return nil, fmt.Errorf("tensor shape contains negative dimension %d", d)
		}
		numel *= d
	}
	if numel < 0 {
		return nil, fmt.Errorf("tensor shape has negative numel")
	}

	cShape := make([]C.int64_t, len(shape))
	cStrides := make([]C.int64_t, len(shape))
	stride := int64(dtype.ItemSize())
	for i := len(shape) - 1; i >= 0; i-- {
		cShape[i] = C.int64_t(shape[i])
		cStrides[i] = C.int64_t(stride)
		stride *= shape[i]
	}
	var shapePtr *C.int64_t
	var stridesPtr *C.int64_t
	if len(cShape) != 0 {
		shapePtr = &cShape[0]
		stridesPtr = &cStrides[0]
	}

	ptr := C.pygo_tensor_create(
		nil,
		shapePtr, stridesPtr,
		C.int(len(shape)),
		C.int(dtype),
		C.int(device),
		C.int64_t(numel),
		C.int(dtype.ItemSize()),
	)
	if ptr == nil {
		return nil, fmt.Errorf("failed to allocate BridgeTensor")
	}

	t := &BridgeTensor{ptr: ptr, shape: append([]int64(nil), shape...), dtype: dtype, device: device}
	runtime.SetFinalizer(t, func(bt *BridgeTensor) { C.pygo_tensor_free(bt.ptr) })
	return t, nil
}

// DataPtr returns the raw unsafe pointer to tensor data.
// Use carefully — the Go GC must not collect the BridgeTensor while this is in use.
func (t *BridgeTensor) DataPtr() unsafe.Pointer {
	return C.pygo_tensor_data(t.ptr)
}

// Shape returns the tensor shape.
func (t *BridgeTensor) Shape() []int64 { return t.shape }

// DType returns the tensor data type.
func (t *BridgeTensor) DType() DType { return t.dtype }

// NumEl returns the total number of elements.
func (t *BridgeTensor) NumEl() int64 {
	n := int64(1)
	for _, d := range t.shape {
		n *= d
	}
	return n
}

// Float32Slice returns a Go slice backed by the tensor data (zero-copy).
// Only valid when DType == Float32.
func (t *BridgeTensor) Float32Slice() []float32 {
	if t.dtype != Float32 {
		panic("Float32Slice called on non-float32 tensor")
	}
	n := t.NumEl()
	return (*[1 << 30]float32)(t.DataPtr())[:n:n]
}

// ToPython wraps this tensor as a Python capsule (zero-copy).
// The capsule can be passed directly to Python functions.
func (t *BridgeTensor) ToPython() unsafe.Pointer {
	return unsafe.Pointer(C.pygo_tensor_to_python(t.ptr))
}

// ─── Python Module Caller ─────────────────────────────────────────────────────

// PyModule wraps an imported Python module.
type PyModule struct{ obj unsafe.Pointer }

// ImportModule imports a Python module by name.
// Must be called with GIL held.
func ImportModule(name string) (*PyModule, error) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	obj := C.pygo_import_module(cname)
	if obj == nil {
		return nil, fmt.Errorf("failed to import Python module '%s': %s", name, getError())
	}
	return &PyModule{obj: unsafe.Pointer(obj)}, nil
}

// CallFunction calls a function in this module with no arguments.
// Must be called with GIL held.
func (m *PyModule) CallFunction(name string) error {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	result := C.pygo_call_function(
		(*C.PyObject)(m.obj), cname, nil, nil,
	)
	if result == nil {
		return fmt.Errorf("Python call '%s' failed: %s", name, getError())
	}
	return nil
}

func getError() string {
	var buf [4096]C.char
	if C.pygo_get_error(&buf[0], C.size_t(len(buf))) == 0 {
		return "unknown error"
	}
	return C.GoString(&buf[0])
}

// ─── High-level helpers ───────────────────────────────────────────────────────

// RunPython runs a Python string expression in the embedded interpreter.
func RunPython(code string) error {
	if err := InitPython(); err != nil {
		return err
	}
	var runErr error
	WithGIL(func() {
		ccode := C.CString(code)
		defer C.free(unsafe.Pointer(ccode))
		if C.pygo_run_string(ccode) != 0 {
			runErr = fmt.Errorf("Python error: %s", getError())
		}
	})
	return runErr
}
