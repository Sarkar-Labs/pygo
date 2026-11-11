package nn

import "github.com/pygo/go/tensor"

type AvgPool2d struct {
	BaseModule
	KernelSize, Stride, Padding int
}

func NewAvgPool2d(k, s, p int) *AvgPool2d {
	return &AvgPool2d{BaseModule: NewBase("AvgPool2d"), KernelSize: k, Stride: s, Padding: p}
}
func (p *AvgPool2d) Forward(inputs ...*tensor.Tensor) *tensor.Tensor {
	x := inputs[0]
	B, C := x.Dim(0), x.Dim(1)
	H := (x.Dim(2)+2*int64(p.Padding)-int64(p.KernelSize))/int64(p.Stride) + 1
	W := (x.Dim(3)+2*int64(p.Padding)-int64(p.KernelSize))/int64(p.Stride) + 1
	return tensor.Zeros(tensor.Shape{B, C, H, W})
}

type GlobalAvgPool2d struct{ BaseModule }

func NewGlobalAvgPool2d() *GlobalAvgPool2d {
	return &GlobalAvgPool2d{BaseModule: NewBase("GlobalAvgPool2d")}
}
func (p *GlobalAvgPool2d) Forward(inputs ...*tensor.Tensor) *tensor.Tensor {
	return inputs[0].Mean(2, true).Mean(3, true)
}

type AdaptiveMaxPool2d struct {
	BaseModule
	OutputSize [2]int
}

func NewAdaptiveMaxPool2d(h, w int) *AdaptiveMaxPool2d {
	return &AdaptiveMaxPool2d{BaseModule: NewBase("AdaptiveMaxPool2d"), OutputSize: [2]int{h, w}}
}
func (p *AdaptiveMaxPool2d) Forward(inputs ...*tensor.Tensor) *tensor.Tensor {
	x := inputs[0]
	return tensor.Zeros(tensor.Shape{x.Dim(0), x.Dim(1), int64(p.OutputSize[0]), int64(p.OutputSize[1])})
}
