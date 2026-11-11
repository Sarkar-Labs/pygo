package tensor

/*
#cgo LDFLAGS: -L${SRCDIR}/../../cpp/release/native -L${SRCDIR}/../../build-release -lpygo -lstdc++ -lm -Wl,-rpath,${SRCDIR}/../../cpp/release/native -Wl,-rpath,${SRCDIR}/../../build-release
#include "../../cpp/bindings/cgo/tensor_cgo.h"
*/
import "C"

import "unsafe"

// CPtr returns the native data pointer. It is only valid while t remains alive
// and currently supports the CPU float32 ABI exposed by PyGo.
func (t *Tensor) CPtr() unsafe.Pointer {
	if t == nil || t.ptr == nil {
		return nil
	}
	return C.PyGoTensor_DataF32(t.ptr)
}

// FromCPtr imports an existing pointer by copying it into native PyGo storage.
// A zero-copy external-buffer constructor belongs to the stable C ABI and is
// intentionally not exposed through a raw Go pointer because Go's cgo pointer
// rules prohibit retaining arbitrary Go memory in C++.
func FromCPtr(ptr unsafe.Pointer, shape Shape, dtype DType, dev Device) *Tensor {
	if dtype != Float32 || dev.Kind != CPU {
		panic("tensor.FromCPtr currently supports CPU float32 only")
	}
	return fromRawF32(ptr, shape, dtype, dev)
}

func (t *Tensor) ByteSlice() []byte {
	if t == nil || t.dtype != Float32 {
		return nil
	}
	p := t.CPtr()
	if p == nil || t.NumEl() == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(p), int(t.NumEl())*4)
}

func (t *Tensor) ShapeInts() []int {
	out := make([]int, len(t.shape))
	for i, v := range t.shape {
		out[i] = int(v)
	}
	return out
}
func (t *Tensor) Itemsize() int { return t.dtype.ItemSize() }
func (t *Tensor) Numel() int64  { return t.NumEl() }

func fromRawF32(ptr unsafe.Pointer, shape Shape, dtype DType, dev Device) *Tensor {
	if ptr == nil && shape.NumEl() != 0 {
		panic("tensor.FromCPtr: nil pointer for non-empty tensor")
	}
	bytes := int(shape.NumEl()) * 4
	h := C.PyGoTensor_FromData(ptr, C.size_t(bytes), (*C.int64_t)(unsafe.Pointer(&shape[0])), C.int(len(shape)), C.int(dtype), C.int(dev.Kind), C.int(dev.Index))
	return wrapHandle(h, shape, dtype, dev)
}
