package nn

import "github.com/pygo/go/tensor"

// Conv2d is the public 2-D convolution module. The stable Go tensor ABI does
// not yet expose a native convolution kernel, so Forward fails explicitly
// rather than returning an incorrect tensor.
type Conv2d struct {
	BaseModule
	InChannels, OutChannels     int
	KernelSize, Stride, Padding int
	BiasEnabled                 bool
	Weight, Bias                *Parameter
}

func NewConv2d(inChannels, outChannels, kernelSize, stride, padding int, bias bool) *Conv2d {
	c := &Conv2d{BaseModule: NewBase("Conv2d"), InChannels: inChannels, OutChannels: outChannels, KernelSize: kernelSize, Stride: stride, Padding: padding, BiasEnabled: bias}
	c.Weight = NewParameter(tensor.Randn(tensor.Shape{int64(outChannels), int64(inChannels), int64(kernelSize), int64(kernelSize)}))
	c.RegisterParameter("weight", c.Weight)
	if bias {
		c.Bias = NewParameter(tensor.Zeros(tensor.Shape{int64(outChannels)}))
		c.RegisterParameter("bias", c.Bias)
	}
	return c
}
func (c *Conv2d) Forward(inputs ...*tensor.Tensor) *tensor.Tensor {
	panic("nn.Conv2d: native convolution kernel is not exposed by the stable Go tensor ABI")
}
