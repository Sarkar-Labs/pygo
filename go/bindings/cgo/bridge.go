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

// ─── Python Runtime ───────────────────────────────────────────────────────────

static int pygo_init_python(const char* program) {
    if (Py_IsInitialized()) return 0;
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
    void*    data_ptr;    // Raw data pointer (shared between Go/Python/C++)
    int64_t* shape;       // Dimension sizes
    int64_t* strides;     // Strides in bytes
    int      ndim;        // Number of dimensions
    int      dtype;       // 0=f32, 1=f16, 2=bf16, 3=f64, 4=i8, 5=i64
    int      device;      // 0=cpu, 1=cuda:0, 2=cuda:1, ...
    int64_t  numel;       // Total number of elements
    int      itemsize;    // Bytes per element
} PyGoTensor;

static PyGoTensor* pygo_tensor_create(
    void* data, int64_t* shape, int64_t* strides,
    int ndim, int dtype, int device, int64_t numel, int itemsize
) {
    PyGoTensor* t = (PyGoTensor*)malloc(sizeof(PyGoTensor));
    if (!t) return NULL;
    t->data_ptr = data;
    t->shape    = (int64_t*)malloc(ndim * sizeof(int64_t));
    t->strides  = (int64_t*)malloc(ndim * sizeof(int64_t));
    if (!t->shape || !t->strides) { free(t); return NULL; }
    memcpy(t->shape,   shape,   ndim * sizeof(int64_t));
    memcpy(t->strides, strides, ndim * sizeof(int64_t));
    t->ndim     = ndim;
    t->dtype    = dtype;
    t->device   = device;
    t->numel    = numel;
    t->itemsize = itemsize;
    return t;
}

static void pygo_tensor_free(PyGoTensor* t) {
    if (!t) return;
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
    PyObject* result = PyObject_Call(fn, args ? args : PyTuple_New(0), kwargs);
    Py_DECREF(fn);
    return result;
}

static const char* pygo_get_error() {
    if (!PyErr_Occurred()) return NULL;
    PyObject *type, *value, *traceback;
    PyErr_Fetch(&type, &value, &traceback);
    PyErr_NormalizeException(&type, &value, &traceback);
    PyObject* str = PyObject_Str(value);
    const char* msg = str ? PyUnicode_AsUTF8(str) : "unknown error";
    Py_XDECREF(type); Py_XDECREF(value); Py_XDECREF(traceback); Py_XDECREF(str);
    return msg;
}

// ─── Tensor → Python capsule (zero-copy) ─────────────────────────────────────

static void _tensor_capsule_destructor(PyObject* cap) {
    // We do NOT free the tensor data here — Go owns the memory
    (void)cap;
}

static PyObject* pygo_tensor_to_python(PyGoTensor* t) {
    if (!t) { Py_RETURN_NONE; }
    return PyCapsule_New(t, "PyGoTensor", _tensor_capsule_destructor);
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
	initOnce sync.Once
	initErr  error
)

// InitPython initializes the embedded CPython interpreter.
// Safe to call multiple times — only initializes once.
func InitPython() error {
	initOnce.Do(func() {
		prog := C.CString(runtime.GOROOT())
		defer C.free(unsafe.Pointer(prog))
		if C.pygo_init_python(prog) != 0 {
			initErr = fmt.Errorf("failed to initialize CPython")
		}
	})
	return initErr
}

// FinalizePython shuts down the embedded CPython interpreter.
func FinalizePython() {
	C.pygo_finalize_python()
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
// The underlying data is allocated by Go (via make([]byte, ...)) and
// its pointer is passed to Python/C++ without copying.
type BridgeTensor struct {
	ptr    *C.PyGoTensor
	data   []byte // Go owns this — prevents GC
	shape  []int64
	dtype  DType
	device int
}

// NewBridgeTensor creates a new tensor with Go-owned memory.
func NewBridgeTensor(shape []int64, dtype DType, device int) (*BridgeTensor, error) {
	numel := int64(1)
	for _, d := range shape {
		numel *= d
	}
	data := make([]byte, numel*int64(dtype.ItemSize()))

	cShape := make([]C.int64_t, len(shape))
	cStrides := make([]C.int64_t, len(shape))
	stride := int64(dtype.ItemSize())
	for i := len(shape) - 1; i >= 0; i-- {
		cShape[i] = C.int64_t(shape[i])
		cStrides[i] = C.int64_t(stride)
		stride *= shape[i]
	}

	ptr := C.pygo_tensor_create(
		unsafe.Pointer(&data[0]),
		&cShape[0], &cStrides[0],
		C.int(len(shape)),
		C.int(dtype),
		C.int(device),
		C.int64_t(numel),
		C.int(dtype.ItemSize()),
	)
	if ptr == nil {
		return nil, fmt.Errorf("failed to allocate BridgeTensor")
	}

	t := &BridgeTensor{ptr: ptr, data: data, shape: shape, dtype: dtype, device: device}
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
	msg := C.pygo_get_error()
	if msg == nil {
		return "unknown error"
	}
	return C.GoString(msg)
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
		if C.PyRun_SimpleString(ccode) != 0 {
			runErr = fmt.Errorf("Python error: %s", getError())
		}
	})
	return runErr
}
