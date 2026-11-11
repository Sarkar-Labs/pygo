//go:build legacy_cgo_bridge
// +build legacy_cgo_bridge

// Package cgo provides the CGo bridge between Go and Python/C++ layers.
// This is the most critical file in PyGo — it enables zero-copy tensor
// sharing between Go, Python (CPython), and C++ compute kernels.
package cgo

/*
#cgo pkg-config: python3-embed
#cgo CFLAGS: -I../../cpp/tensor


#include <Python.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// ─── Python Runtime ───────────────────────────────────────────────────────────

static int pygo_init_python(const char* program) {
    if (Py_IsInitialized()) return 0;
    wchar_t* prog = Py_DecodeLocale(program, NULL);
    if (prog == NULL) return -1;
    Py_SetProgramName(prog);
    Py_Initialize();
    PyMem_RawFree(prog);
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
    void*    data_ptr;
    int64_t* shape;
    int64_t* strides;
    int      ndim;
    int      dtype;
    int      device;
    int64_t  numel;
    int      itemsize;
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
    free(t->shape);
    free(t->strides);
    free(t);
}

static void* pygo_tensor_data(PyGoTensor* t) {
    return t ? t->data_ptr : NULL;
}

static int64_t pygo_tensor_shape_dim(PyGoTensor* t, int dim) {
    if (!t || dim < 0 || dim >= t->ndim) return -1;
    return t->shape[dim];
}

static int64_t pygo_tensor_stride_dim(PyGoTensor* t, int dim) {
    if (!t || dim < 0 || dim >= t->ndim) return -1;
    return t->strides[dim];
}

// ─── Python Capsule (zero-copy tensor to Python) ──────────────────────────────

static void _capsule_destructor(PyObject* cap) { (void)cap; }

static PyObject* pygo_tensor_to_capsule(PyGoTensor* t) {
    if (!t) { Py_RETURN_NONE; }
    return PyCapsule_New((void*)t, "PyGoTensor", _capsule_destructor);
}

static PyGoTensor* pygo_tensor_from_capsule(PyObject* cap) {
    if (!cap || !PyCapsule_IsValid(cap, "PyGoTensor")) return NULL;
    return (PyGoTensor*)PyCapsule_GetPointer(cap, "PyGoTensor");
}

// ─── Module / Function Calls ──────────────────────────────────────────────────

static PyObject* pygo_import_module(const char* name) {
    return PyImport_ImportModule(name);
}

static PyObject* pygo_get_attr(PyObject* obj, const char* name) {
    return PyObject_GetAttrString(obj, name);
}

static PyObject* pygo_call(PyObject* callable, PyObject* args, PyObject* kwargs) {
    if (!callable || !PyCallable_Check(callable)) return NULL;
    return PyObject_Call(callable, args ? args : PyTuple_New(0), kwargs);
}

static int pygo_run_string(const char* code) {
    return PyRun_SimpleString(code);
}

// ─── Error Handling ───────────────────────────────────────────────────────────

static const char* pygo_last_error() {
    if (!PyErr_Occurred()) return NULL;
    PyObject *type, *value, *tb;
    PyErr_Fetch(&type, &value, &tb);
    PyErr_NormalizeException(&type, &value, &tb);
    PyObject* s = value ? PyObject_Str(value) : NULL;
    const char* msg = s ? PyUnicode_AsUTF8(s) : "unknown Python error";
    Py_XDECREF(type); Py_XDECREF(value); Py_XDECREF(tb); Py_XDECREF(s);
    return msg;
}

// ─── GIL ─────────────────────────────────────────────────────────────────────

static PyGILState_STATE pygo_gil_acquire() { return PyGILState_Ensure(); }
static void pygo_gil_release(PyGILState_STATE s) { PyGILState_Release(s); }
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
		prog := C.CString("pygo")
		defer C.free(unsafe.Pointer(prog))
		if C.pygo_init_python(prog) != 0 {
			initErr = fmt.Errorf("failed to initialize CPython interpreter")
		}
	})
	return initErr
}

// FinalizePython shuts down the embedded CPython interpreter.
func FinalizePython() { C.pygo_finalize_python() }

// IsInitialized returns true if CPython is running.
func IsInitialized() bool { return C.pygo_is_initialized() != 0 }

// ─── GIL ─────────────────────────────────────────────────────────────────────

// GILState holds the Python GIL.
type GILState struct{ state C.PyGILState_STATE }

// AcquireGIL acquires the Python GIL. Must call Release() when done.
func AcquireGIL() *GILState { return &GILState{C.pygo_gil_acquire()} }

// Release releases the Python GIL.
func (g *GILState) Release() { C.pygo_gil_release(g.state) }

// WithGIL runs fn while holding the Python GIL.
// This is the recommended way to call Python from Go goroutines.
func WithGIL(fn func()) {
	g := AcquireGIL()
	defer g.Release()
	fn()
}

// ─── DType ───────────────────────────────────────────────────────────────────

// DType represents tensor element types.
type DType int

const (
	Float32  DType = 0
	Float16  DType = 1
	BFloat16 DType = 2
	Float64  DType = 3
	Int8     DType = 4
	Int64    DType = 5
	Int32    DType = 6
	UInt8    DType = 7
)

func (d DType) ItemSize() int {
	switch d {
	case Float32, Int32:
		return 4
	case Float16, BFloat16:
		return 2
	case Float64, Int64:
		return 8
	case Int8, UInt8:
		return 1
	default:
		return 4
	}
}

func (d DType) String() string {
	return [...]string{"float32", "float16", "bfloat16", "float64", "int8", "int64", "int32", "uint8"}[d]
}

// ─── BridgeTensor ────────────────────────────────────────────────────────────

// BridgeTensor is a zero-copy tensor shared between Go, Python, and C++.
// Go owns the underlying memory — Python/C++ get raw pointers via capsule.
type BridgeTensor struct {
	cptr    *C.PyGoTensor
	data    []byte // Go-owned backing store — keeps memory alive
	shape   []int64
	strides []int64
	dtype   DType
	device  int
}

// NewBridgeTensor allocates a new tensor with Go-managed memory.
// device: 0=CPU, 1=CUDA:0, 2=CUDA:1, ...
func NewBridgeTensor(shape []int64, dtype DType, device int) (*BridgeTensor, error) {
	numel := int64(1)
	for _, d := range shape {
		numel *= d
	}
	nbytes := numel * int64(dtype.ItemSize())
	data := make([]byte, nbytes)

	ndim := len(shape)
	cShape := make([]C.int64_t, ndim)
	cStrides := make([]C.int64_t, ndim)
	strides := make([]int64, ndim)

	stride := int64(dtype.ItemSize())
	for i := ndim - 1; i >= 0; i-- {
		cShape[i] = C.int64_t(shape[i])
		cStrides[i] = C.int64_t(stride)
		strides[i] = stride
		stride *= shape[i]
	}

	cptr := C.pygo_tensor_create(
		unsafe.Pointer(&data[0]),
		&cShape[0], &cStrides[0],
		C.int(ndim), C.int(dtype), C.int(device),
		C.int64_t(numel), C.int(dtype.ItemSize()),
	)
	if cptr == nil {
		return nil, fmt.Errorf("BridgeTensor: C allocation failed (shape=%v dtype=%v)", shape, dtype)
	}

	t := &BridgeTensor{
		cptr: cptr, data: data,
		shape: shape, strides: strides,
		dtype: dtype, device: device,
	}
	runtime.SetFinalizer(t, func(bt *BridgeTensor) {
		C.pygo_tensor_free(bt.cptr)
	})
	return t, nil
}

// Shape returns tensor dimensions.
func (t *BridgeTensor) Shape() []int64 { return t.shape }

// Strides returns tensor strides in bytes.
func (t *BridgeTensor) Strides() []int64 { return t.strides }

// DType returns element dtype.
func (t *BridgeTensor) DType() DType { return t.dtype }

// Device returns device index (0=CPU).
func (t *BridgeTensor) Device() int { return t.device }

// NDim returns number of dimensions.
func (t *BridgeTensor) NDim() int { return len(t.shape) }

// NumEl returns total number of elements.
func (t *BridgeTensor) NumEl() int64 {
	n := int64(1)
	for _, d := range t.shape {
		n *= d
	}
	return n
}

// DataPtr returns raw unsafe pointer to the tensor data.
// Pin the BridgeTensor to prevent GC while pointer is in use.
func (t *BridgeTensor) DataPtr() unsafe.Pointer {
	return C.pygo_tensor_data(t.cptr)
}

// Float32Slice returns a Go []float32 backed by tensor data (zero-copy).
// Panics if dtype != Float32.
func (t *BridgeTensor) Float32Slice() []float32 {
	if t.dtype != Float32 {
		panic(fmt.Sprintf("Float32Slice: tensor dtype is %v, expected float32", t.dtype))
	}
	n := t.NumEl()
	ptr := (*float32)(t.DataPtr())
	return (*[1 << 28]float32)(unsafe.Pointer(ptr))[:n:n]
}

// Int64Slice returns a Go []int64 backed by tensor data (zero-copy).
func (t *BridgeTensor) Int64Slice() []int64 {
	if t.dtype != Int64 {
		panic(fmt.Sprintf("Int64Slice: tensor dtype is %v, expected int64", t.dtype))
	}
	n := t.NumEl()
	ptr := (*int64)(t.DataPtr())
	return (*[1 << 28]int64)(unsafe.Pointer(ptr))[:n:n]
}

// ByteSlice returns the raw byte backing store.
func (t *BridgeTensor) ByteSlice() []byte { return t.data }

// ToPython wraps this tensor as a Python capsule.
// The capsule can be passed directly into Python functions.
// Zero-copy: Python gets a raw pointer to Go memory.
func (t *BridgeTensor) ToPython() *C.PyObject {
	return C.pygo_tensor_to_capsule(t.cptr)
}

// FromPython unwraps a Python capsule back to a BridgeTensor.
func FromPython(obj *C.PyObject) *BridgeTensor {
	cptr := C.pygo_tensor_from_capsule(obj)
	if cptr == nil {
		return nil
	}
	ndim := int(cptr.ndim)
	shape := make([]int64, ndim)
	strides := make([]int64, ndim)
	for i := 0; i < ndim; i++ {
		shape[i] = int64(C.pygo_tensor_shape_dim(cptr, C.int(i)))
		strides[i] = int64(C.pygo_tensor_stride_dim(cptr, C.int(i)))
	}
	return &BridgeTensor{
		cptr: cptr, shape: shape, strides: strides,
		dtype: DType(cptr.dtype), device: int(cptr.device),
	}
}

func (t *BridgeTensor) String() string {
	return fmt.Sprintf("BridgeTensor(shape=%v dtype=%v device=%d)", t.shape, t.dtype, t.device)
}

// ─── Python Module ────────────────────────────────────────────────────────────

// Module wraps an imported Python module.
type Module struct{ obj *C.PyObject }

// Import imports a Python module. Must hold GIL.
func Import(name string) (*Module, error) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	obj := C.pygo_import_module(cname)
	if obj == nil {
		return nil, fmt.Errorf("import %q failed: %s", name, lastError())
	}
	return &Module{obj: obj}, nil
}

// Call calls a named function in this module.
func (m *Module) Call(fn string, args ...*C.PyObject) (*C.PyObject, error) {
	cfn := C.CString(fn)
	defer C.free(unsafe.Pointer(cfn))
	callable := C.pygo_get_attr(m.obj, cfn)
	if callable == nil {
		return nil, fmt.Errorf("module has no attr %q", fn)
	}
	var cargs *C.PyObject
	if len(args) > 0 {
		cargs = args[0]
	}
	result := C.pygo_call(callable, cargs, nil)
	if result == nil {
		return nil, fmt.Errorf("call %q failed: %s", fn, lastError())
	}
	return result, nil
}

// ─── Utilities ────────────────────────────────────────────────────────────────

// RunString executes a Python code string in the embedded interpreter.
func RunString(code string) error {
	if err := InitPython(); err != nil {
		return err
	}
	var runErr error
	WithGIL(func() {
		ccode := C.CString(code)
		defer C.free(unsafe.Pointer(ccode))
		if C.pygo_run_string(ccode) != 0 {
			runErr = fmt.Errorf("Python error: %s", lastError())
		}
	})
	return runErr
}

func lastError() string {
	msg := C.pygo_last_error()
	if msg == nil {
		return "unknown error"
	}
	return C.GoString(msg)
}
