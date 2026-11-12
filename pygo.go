// Package pygo is the public Go-first entry point for PyGo.
//
// Users write Go applications against this package and its public Go
// subpackages (tensor, nn, optim, training, models, inference, ...).
// Python and C++/CUDA are implementation backends behind the Go API; they are
// not required as user-facing programming languages.
package pygo

import "github.com/Sarkar-Labs/pygo/go/tensor"

// Re-export the core tensor types so small programs can use PyGo without
// importing an implementation subpackage.
type Tensor = tensor.Tensor
type Shape = tensor.Shape
type DType = tensor.DType
type Device = tensor.Device
type DeviceKind = tensor.DeviceKind
type Option = tensor.Option

const (
	Float32   = tensor.Float32
	Float16   = tensor.Float16
	BFloat16  = tensor.BFloat16
	Float64   = tensor.Float64
	Int8      = tensor.Int8
	Int16     = tensor.Int16
	Int32     = tensor.Int32
	Int64     = tensor.Int64
	UInt8     = tensor.UInt8
	Bool      = tensor.Bool
	Complex64 = tensor.Complex64
	Int4      = tensor.Int4
	NFloat4   = tensor.NFloat4
)

const (
	CPU    = tensor.CPU
	CUDA   = tensor.CUDA
	MPS    = tensor.MPS
	XLA    = tensor.XLA
	OpenCL = tensor.OpenCL
)

var (
	DevCPU = tensor.DevCPU
	DevMPS = tensor.DevMPS
)

func DevCUDA(index int) Device                   { return tensor.DevCUDA(index) }
func Zeros(shape Shape, opts ...Option) *Tensor { return tensor.Zeros(shape, opts...) }
func Ones(shape Shape, opts ...Option) *Tensor  { return tensor.Ones(shape, opts...) }
func Randn(shape Shape, opts ...Option) *Tensor { return tensor.Randn(shape, opts...) }

func Arange(start, stop, step float64, opts ...Option) *Tensor {
	return tensor.Arange(start, stop, step, opts...)
}

func FromSlice[T float32 | float64 | int32 | int64 | uint8](
	values []T,
	shape Shape,
	opts ...Option,
) *Tensor {
	return tensor.FromSlice(values, shape, opts...)
}

func WithDType(dt DType) Option  { return tensor.WithDType(dt) }
func WithDevice(d Device) Option { return tensor.WithDevice(d) }
func OnCPU() Option              { return tensor.OnCPU() }
func OnCUDA(index int) Option    { return tensor.OnCUDA(index) }
