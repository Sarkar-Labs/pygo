// Package tensor provides the core N-dimensional array type for PyGo.
// Architecture: Go API → CGo → Python (autograd) → C++ (compute kernel)
package tensor

/*
#cgo LDFLAGS: -L${SRCDIR}/../../cpp/release/native -L${SRCDIR}/../../build-release -lpygo -lstdc++ -lm -Wl,-rpath,${SRCDIR}/../../cpp/release/native -Wl,-rpath,${SRCDIR}/../../build-release
#cgo CXXFLAGS: -std=c++20
#include "../../cpp/bindings/cgo/tensor_cgo.h"
#include <stdlib.h>
*/
import "C"

import (
	"fmt"
	"math"
	"runtime"
	"unsafe"
)

// ─── DType ───────────────────────────────────────────────────────────────────

// DType represents element data types.
type DType int8

const (
	Float32   DType = 0
	Float16   DType = 1
	BFloat16  DType = 2
	Float64   DType = 3
	Int8      DType = 4
	Int16     DType = 5
	Int32     DType = 6
	Int64     DType = 7
	UInt8     DType = 8
	Bool      DType = 9
	Complex64 DType = 10
	Int4      DType = 12
	NFloat4   DType = 13
)

func (d DType) String() string {
	names := map[DType]string{
		Float32: "float32", Float16: "float16", BFloat16: "bfloat16",
		Float64: "float64", Int8: "int8", Int16: "int16",
		Int32: "int32", Int64: "int64", UInt8: "uint8",
		Bool: "bool", Int4: "int4", NFloat4: "nfloat4",
	}
	if s, ok := names[d]; ok {
		return s
	}
	return "unknown"
}

func (d DType) ItemSize() int {
	sizes := map[DType]int{
		Float32: 4, Float16: 2, BFloat16: 2, Float64: 8,
		Int8: 1, Int16: 2, Int32: 4, Int64: 8, UInt8: 1,
		Bool: 1, Int4: 1, NFloat4: 1,
	}
	return sizes[d]
}

// ─── Device ──────────────────────────────────────────────────────────────────

// Device represents where a tensor lives.
type Device struct {
	Kind  DeviceKind
	Index int // GPU index
}

type DeviceKind int8

const (
	CPU    DeviceKind = 0
	CUDA   DeviceKind = 1
	MPS    DeviceKind = 2 // Apple Metal
	XLA    DeviceKind = 3
	OpenCL DeviceKind = 4
)

var (
	DevCPU  = Device{CPU, 0}
	DevCUDA = func(i int) Device { return Device{CUDA, i} }
	DevMPS  = Device{MPS, 0}
)

func (d Device) String() string {
	switch d.Kind {
	case CPU:
		return "cpu"
	case CUDA:
		return fmt.Sprintf("cuda:%d", d.Index)
	case MPS:
		return "mps"
	case XLA:
		return fmt.Sprintf("xla:%d", d.Index)
	}
	return "unknown"
}

// ─── Shape ───────────────────────────────────────────────────────────────────

// Shape is an immutable list of dimension sizes.
type Shape []int64

func (s Shape) NDim() int {
	return len(s)
}

func (s Shape) NumEl() int64 {
	if len(s) == 0 {
		return 0
	}
	n := int64(1)
	for _, d := range s {
		n *= d
	}
	return n
}

func (s Shape) String() string {
	return fmt.Sprintf("%v", []int64(s))
}

func (s Shape) Equals(o Shape) bool {
	if len(s) != len(o) {
		return false
	}
	for i := range s {
		if s[i] != o[i] {
			return false
		}
	}
	return true
}

// ─── Tensor ──────────────────────────────────────────────────────────────────

// Tensor is PyGo's primary n-dimensional array.
// The underlying data lives in C++ (possibly on GPU); Go holds a reference.
type Tensor struct {
	ptr    C.PyGoTensorHandle // opaque pointer to C++ TensorImpl
	shape  Shape
	dtype  DType
	device Device
	// autograd fields
	requiresGrad bool
	gradFn       GradFn
}

// GradFn is a Go interface for autograd nodes.
type GradFn interface {
	Apply(grads []*Tensor) []*Tensor
	Name() string
}

// finalizer: release C++ memory when Go GC collects this tensor.
func setFinalizer(t *Tensor) {
	runtime.SetFinalizer(t, func(tt *Tensor) {
		if tt.ptr != nil {
			C.PyGoTensor_Release(tt.ptr)
			tt.ptr = nil
		}
	})
}

// ── Properties ──────────────────────────────────────────────────────────────

func (t *Tensor) Shape() Shape   { return t.shape }
func (t *Tensor) NDim() int      { return len(t.shape) }
func (t *Tensor) NumEl() int64   { return t.shape.NumEl() }
func (t *Tensor) DType() DType   { return t.dtype }
func (t *Tensor) Device() Device { return t.device }
func (t *Tensor) Dim(i int) int64 {
	if i < 0 {
		i += len(t.shape)
	}
	return t.shape[i]
}
func (t *Tensor) RequiresGrad() bool { return t.requiresGrad }
func (t *Tensor) IsLeaf() bool       { return t.gradFn == nil }

// ── Factory helpers (calls C++ kernels) ─────────────────────────────────────

func Zeros(shape Shape, opts ...Option) *Tensor {
	o := applyOptions(opts)
	h := C.PyGoTensor_Zeros(
		(*C.int64_t)(unsafe.Pointer(&shape[0])), C.int(len(shape)),
		C.int(o.dtype), C.int(o.device.Kind), C.int(o.device.Index),
	)
	t := &Tensor{ptr: h, shape: shape, dtype: o.dtype, device: o.device}
	setFinalizer(t)
	return t
}

func Ones(shape Shape, opts ...Option) *Tensor {
	o := applyOptions(opts)
	h := C.PyGoTensor_Ones(
		(*C.int64_t)(unsafe.Pointer(&shape[0])), C.int(len(shape)),
		C.int(o.dtype), C.int(o.device.Kind), C.int(o.device.Index),
	)
	t := &Tensor{ptr: h, shape: shape, dtype: o.dtype, device: o.device}
	setFinalizer(t)
	return t
}

func Randn(shape Shape, opts ...Option) *Tensor {
	o := applyOptions(opts)
	h := C.PyGoTensor_Randn(
		(*C.int64_t)(unsafe.Pointer(&shape[0])), C.int(len(shape)),
		C.int(o.dtype), C.int(o.device.Kind), C.int(o.device.Index),
	)
	t := &Tensor{ptr: h, shape: shape, dtype: o.dtype, device: o.device}
	setFinalizer(t)
	return t
}

func Arange(start, stop, step float64, opts ...Option) *Tensor {
	o := applyOptions(opts)
	n := int64(math.Ceil((stop - start) / step))
	shp := Shape{n}
	h := C.PyGoTensor_Arange(C.double(start), C.double(stop), C.double(step),
		C.int(o.dtype), C.int(o.device.Kind), C.int(o.device.Index))
	t := &Tensor{ptr: h, shape: shp, dtype: o.dtype, device: o.device}
	setFinalizer(t)
	return t
}

func FromSlice[T float32 | float64 | int32 | int64 | uint8](
	data []T, shape Shape, opts ...Option) *Tensor {
	o := applyOptions(opts)
	h := C.PyGoTensor_FromData(
		unsafe.Pointer(&data[0]), C.size_t(len(data)*int(unsafe.Sizeof(data[0]))),
		(*C.int64_t)(unsafe.Pointer(&shape[0])), C.int(len(shape)),
		C.int(o.dtype), C.int(o.device.Kind), C.int(o.device.Index),
	)
	t := &Tensor{ptr: h, shape: shape, dtype: o.dtype, device: o.device}
	setFinalizer(t)
	return t
}

// ── Math ops ──────────────────────────────────────────────────────────────────

func (t *Tensor) Add(other *Tensor) *Tensor    { return binOp(t, other, C.PyGoOp_Add) }
func (t *Tensor) Sub(other *Tensor) *Tensor    { return binOp(t, other, C.PyGoOp_Sub) }
func (t *Tensor) Mul(other *Tensor) *Tensor    { return binOp(t, other, C.PyGoOp_Mul) }
func (t *Tensor) Div(other *Tensor) *Tensor    { return binOp(t, other, C.PyGoOp_Div) }
func (t *Tensor) MatMul(other *Tensor) *Tensor { return binOp(t, other, C.PyGoOp_MatMul) }

func (t *Tensor) AddScalar(v float64) *Tensor {
	h := C.PyGoTensor_AddScalar(t.ptr, C.double(v))
	return wrapHandle(h, t.shape, t.dtype, t.device)
}

func (t *Tensor) MulScalar(v float64) *Tensor {
	h := C.PyGoTensor_MulScalar(t.ptr, C.double(v))
	return wrapHandle(h, t.shape, t.dtype, t.device)
}

func (t *Tensor) Pow(exp float64) *Tensor {
	h := C.PyGoTensor_Pow(t.ptr, C.double(exp))
	return wrapHandle(h, t.shape, t.dtype, t.device)
}

func (t *Tensor) Sqrt() *Tensor    { return unaryOp(t, C.PyGoOp_Sqrt) }
func (t *Tensor) Abs() *Tensor     { return unaryOp(t, C.PyGoOp_Abs) }
func (t *Tensor) Exp() *Tensor     { return unaryOp(t, C.PyGoOp_Exp) }
func (t *Tensor) Log() *Tensor     { return unaryOp(t, C.PyGoOp_Log) }
func (t *Tensor) Sin() *Tensor     { return unaryOp(t, C.PyGoOp_Sin) }
func (t *Tensor) Cos() *Tensor     { return unaryOp(t, C.PyGoOp_Cos) }
func (t *Tensor) Tanh() *Tensor    { return unaryOp(t, C.PyGoOp_Tanh) }
func (t *Tensor) Neg() *Tensor     { return unaryOp(t, C.PyGoOp_Neg) }
func (t *Tensor) ReLU() *Tensor    { return unaryOp(t, C.PyGoOp_ReLU) }
func (t *Tensor) GELU() *Tensor    { return unaryOp(t, C.PyGoOp_GELU) }
func (t *Tensor) SiLU() *Tensor    { return unaryOp(t, C.PyGoOp_SiLU) }
func (t *Tensor) Sigmoid() *Tensor { return unaryOp(t, C.PyGoOp_Sigmoid) }

func (t *Tensor) Floor() *Tensor {
	return mapFloat32Unary(t, func(v float32) float32 { return float32(math.Floor(float64(v))) })
}
func (t *Tensor) Ceil() *Tensor {
	return mapFloat32Unary(t, func(v float32) float32 { return float32(math.Ceil(float64(v))) })
}
func (t *Tensor) Round() *Tensor {
	return mapFloat32Unary(t, func(v float32) float32 { return float32(math.Round(float64(v))) })
}
func (t *Tensor) Log2() *Tensor {
	return mapFloat32Unary(t, func(v float32) float32 { return float32(math.Log2(float64(v))) })
}
func (t *Tensor) Log10() *Tensor {
	return mapFloat32Unary(t, func(v float32) float32 { return float32(math.Log10(float64(v))) })
}

func mapFloat32Unary(t *Tensor, fn func(float32) float32) *Tensor {
	if t == nil {
		return nil
	}
	if t.dtype != Float32 {
		return t.Clone()
	}
	out := t.Clone()
	data := out.nativeFloat32Slice()
	for i, v := range data {
		data[i] = fn(v)
	}
	return out
}

func (t *Tensor) Softmax(dim int) *Tensor {
	h := C.PyGoTensor_Softmax(t.ptr, C.int(dim))
	return wrapHandle(h, t.shape, t.dtype, t.device)
}

// ── Reduction ─────────────────────────────────────────────────────────────────

func (t *Tensor) Sum(dim int, keepdim bool) *Tensor {
	h := C.PyGoTensor_ReduceSum(t.ptr, C.int(dim), C.bool(keepdim))
	return wrapHandleInfer(h, t, dim, keepdim)
}

func (t *Tensor) Mean(dim int, keepdim bool) *Tensor {
	h := C.PyGoTensor_ReduceMean(t.ptr, C.int(dim), C.bool(keepdim))
	return wrapHandleInfer(h, t, dim, keepdim)
}

func (t *Tensor) Max(dim int, keepdim bool) *Tensor {
	h := C.PyGoTensor_ReduceMax(t.ptr, C.int(dim), C.bool(keepdim))
	return wrapHandleInfer(h, t, dim, keepdim)
}

func (t *Tensor) Argmax(dim int) *Tensor {
	h := C.PyGoTensor_Argmax(t.ptr, C.int(dim))
	return wrapHandleUnknownShape(h, Int64, t.device)
}

func (t *Tensor) Item() float64 {
	if t.NumEl() != 1 {
		panic("item() only valid on scalar tensors")
	}
	return float64(C.PyGoTensor_Item(t.ptr))
}

// ── Shape ops ─────────────────────────────────────────────────────────────────

func (t *Tensor) Reshape(shape Shape) *Tensor {
	if shape.NumEl() != t.NumEl() {
		panic(fmt.Sprintf("reshape: size mismatch %v → %v", t.shape, shape))
	}
	h := C.PyGoTensor_Reshape(
		t.ptr, (*C.int64_t)(unsafe.Pointer(&shape[0])), C.int(len(shape)))
	return wrapHandle(h, shape, t.dtype, t.device)
}

func (t *Tensor) Flatten(start, end int) *Tensor {
	h := C.PyGoTensor_Flatten(t.ptr, C.int(start), C.int(end))
	return wrapHandleUnknownShape(h, t.dtype, t.device)
}

func (t *Tensor) Transpose(d0, d1 int) *Tensor {
	newShape := make(Shape, len(t.shape))
	copy(newShape, t.shape)
	newShape[d0], newShape[d1] = newShape[d1], newShape[d0]
	h := C.PyGoTensor_Transpose(t.ptr, C.int(d0), C.int(d1))
	return wrapHandle(h, newShape, t.dtype, t.device)
}

func (t *Tensor) Permute(dims []int) *Tensor {
	newShape := make(Shape, len(t.shape))
	for i, d := range dims {
		newShape[i] = t.shape[d]
	}
	cDims := make([]C.int, len(dims))
	for i, d := range dims {
		cDims[i] = C.int(d)
	}
	h := C.PyGoTensor_Permute(t.ptr, &cDims[0], C.int(len(dims)))
	return wrapHandle(h, newShape, t.dtype, t.device)
}

func (t *Tensor) Squeeze(dim int) *Tensor {
	h := C.PyGoTensor_Squeeze(t.ptr, C.int(dim))
	return wrapHandleUnknownShape(h, t.dtype, t.device)
}

func (t *Tensor) Unsqueeze(dim int) *Tensor {
	newShape := make(Shape, len(t.shape)+1)
	if dim < 0 {
		dim += len(t.shape) + 1
	}
	copy(newShape[:dim], t.shape[:dim])
	newShape[dim] = 1
	copy(newShape[dim+1:], t.shape[dim:])
	h := C.PyGoTensor_Unsqueeze(t.ptr, C.int(dim))
	return wrapHandle(h, newShape, t.dtype, t.device)
}

// ── Device ops ────────────────────────────────────────────────────────────────

func (t *Tensor) To(dev Device) *Tensor {
	if t.device == dev {
		return t
	}
	h := C.PyGoTensor_To(t.ptr, C.int(dev.Kind), C.int(dev.Index))
	return wrapHandle(h, t.shape, t.dtype, dev)
}

func (t *Tensor) CUDA(idx int) *Tensor { return t.To(DevCUDA(idx)) }
func (t *Tensor) CPU() *Tensor         { return t.To(DevCPU) }
func (t *Tensor) MPS() *Tensor         { return t.To(DevMPS) }

func (t *Tensor) CastTo(dt DType) *Tensor {
	h := C.PyGoTensor_Cast(t.ptr, C.int(dt))
	return wrapHandle(h, t.shape, dt, t.device)
}

func (t *Tensor) Half() *Tensor     { return t.CastTo(Float16) }
func (t *Tensor) BFloat16() *Tensor { return t.CastTo(BFloat16) }
func (t *Tensor) Float() *Tensor    { return t.CastTo(Float32) }

// ── Autograd ──────────────────────────────────────────────────────────────────

func (t *Tensor) Backward(grad *Tensor, retainGraph bool) {
	var gptr C.PyGoTensorHandle
	if grad != nil {
		gptr = grad.ptr
	}
	C.PyGoTensor_Backward(t.ptr, gptr, C.bool(retainGraph))
}

func (t *Tensor) SetRequiresGrad(v bool) *Tensor {
	C.PyGoTensor_SetRequiresGrad(t.ptr, C.bool(v))
	t.requiresGrad = v
	return t
}

func (t *Tensor) Grad() *Tensor {
	h := C.PyGoTensor_GetGrad(t.ptr)
	if h == nil {
		return nil
	}
	return wrapHandleUnknownShape(h, t.dtype, t.device)
}

func (t *Tensor) ZeroGrad() {
	C.PyGoTensor_ZeroGrad(t.ptr)
}

func (t *Tensor) Detach() *Tensor {
	h := C.PyGoTensor_Detach(t.ptr)
	return wrapHandle(h, t.shape, t.dtype, t.device)
}

// ── Clone / Contiguous ───────────────────────────────────────────────────────

func (t *Tensor) Clone() *Tensor {
	h := C.PyGoTensor_Clone(t.ptr)
	return wrapHandle(h, t.shape, t.dtype, t.device)
}

func (t *Tensor) Contiguous() *Tensor {
	h := C.PyGoTensor_Contiguous(t.ptr)
	return wrapHandle(h, t.shape, t.dtype, t.device)
}

// ── String ────────────────────────────────────────────────────────────────────

func (t *Tensor) String() string {
	cstr := C.PyGoTensor_ToString(t.ptr, 16)
	defer C.free(unsafe.Pointer(cstr))
	return fmt.Sprintf("tensor(shape=%v dtype=%s device=%s\n%s)",
		t.shape, t.dtype, t.device, C.GoString(cstr))
}

// ─── internal helpers ────────────────────────────────────────────────────────

func binOp(a, b *Tensor, op C.PyGoOpCode) *Tensor {
	h := C.PyGoTensor_BinOp(a.ptr, b.ptr, op)
	return wrapHandleUnknownShape(h, a.dtype, a.device)
}

func unaryOp(a *Tensor, op C.PyGoOpCode) *Tensor {
	h := C.PyGoTensor_UnaryOp(a.ptr, op)
	return wrapHandle(h, a.shape, a.dtype, a.device)
}

func wrapHandle(h C.PyGoTensorHandle, shape Shape, dt DType, dev Device) *Tensor {
	t := &Tensor{ptr: h, shape: shape, dtype: dt, device: dev}
	setFinalizer(t)
	return t
}

func wrapHandleUnknownShape(h C.PyGoTensorHandle, dt DType, dev Device) *Tensor {
	ndim := int(C.PyGoTensor_NDim(h))
	shape := make(Shape, ndim)
	C.PyGoTensor_Shape(h, (*C.int64_t)(unsafe.Pointer(&shape[0])))
	return wrapHandle(h, shape, dt, dev)
}

func wrapHandleInfer(h C.PyGoTensorHandle, src *Tensor, dim int, keepdim bool) *Tensor {
	return wrapHandleUnknownShape(h, src.dtype, src.device)
}

// ─── Options ─────────────────────────────────────────────────────────────────

type options struct {
	dtype  DType
	device Device
}

type Option func(*options)

func WithDType(dt DType) Option  { return func(o *options) { o.dtype = dt } }
func WithDevice(d Device) Option { return func(o *options) { o.device = d } }
func OnCUDA(idx int) Option      { return WithDevice(DevCUDA(idx)) }
func OnCPU() Option              { return WithDevice(DevCPU) }

func applyOptions(opts []Option) options {
	o := options{dtype: Float32, device: DevCPU}
	for _, fn := range opts {
		fn(&o)
	}
	return o
}

// The following advanced operations are kept behind explicit unsupported errors
// until their corresponding stable C ABI entry points are added.
func (t *Tensor) nativeSlice(dim int, start, end, step int64) *Tensor {
	panic("tensor.Slice: stable C ABI support is not implemented")
}
func (t *Tensor) nativeNorm(p float64, dim int, keepdim bool) *Tensor {
	panic("tensor.Norm: stable C ABI support is not implemented")
}
func (t *Tensor) nativeTriu(diagonal int) *Tensor {
	panic("tensor.Triu: stable C ABI support is not implemented")
}
func (t *Tensor) nativeIndex(idx *Tensor) *Tensor {
	panic("tensor.Index: stable C ABI support is not implemented")
}
func (t *Tensor) nativeGather(dim int, idx *Tensor) *Tensor {
	panic("tensor.Gather: stable C ABI support is not implemented")
}
func (t *Tensor) nativeExpand(sizes ...int64) *Tensor {
	panic("tensor.Expand: stable C ABI support is not implemented")
}
func (t *Tensor) nativeFloat32Slice() []float32 {
	if t == nil || t.ptr == nil {
		return nil
	}
	if t.dtype != Float32 {
		panic("tensor.Float32Slice: tensor dtype must be float32")
	}
	ptr := C.PyGoTensor_DataF32(t.ptr)
	if ptr == nil || t.NumEl() == 0 {
		return nil
	}
	n := int(t.NumEl())
	return unsafe.Slice((*float32)(ptr), n)
}
func (t *Tensor) nativeInt64Slice() []int64 {
	panic("tensor.Int64Slice: stable C ABI data export is not implemented")
}

func (t *Tensor) nativeSign() *Tensor { panic("tensor.Sign: stable C ABI support is not implemented") }
func (t *Tensor) nativeRepeat(repeats []int) *Tensor {
	panic("tensor.Repeat: stable C ABI support is not implemented")
}
