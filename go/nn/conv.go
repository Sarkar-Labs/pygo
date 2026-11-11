package nn

import "github.com/pygo/go/tensor"

type DepthwiseSeparableConv2d struct {
	BaseModule
	DW, PW   *Conv2d
	BN1, BN2 *BatchNorm2d
}

func NewDepthwiseSeparableConv2d(in, out, k, s, p int) *DepthwiseSeparableConv2d {
	d := &DepthwiseSeparableConv2d{BaseModule: NewBase("DWSep")}
	d.DW = NewConv2d(in, in, k, s, p, false)
	d.PW = NewConv2d(in, out, 1, 1, 0, false)
	d.BN1 = NewBatchNorm2d(in, 1e-5)
	d.BN2 = NewBatchNorm2d(out, 1e-5)
	d.RegisterModule("dw", d.DW)
	d.RegisterModule("pw", d.PW)
	return d
}
func (d *DepthwiseSeparableConv2d) Forward(inputs ...*tensor.Tensor) *tensor.Tensor {
	return d.BN2.Forward(d.PW.Forward(d.BN1.Forward(d.DW.Forward(inputs[0])).ReLU())).ReLU()
}

type ConvBNReLU struct {
	BaseModule
	Conv    *Conv2d
	BN      *BatchNorm2d
	UseReLU bool
}

func NewConvBNReLU(in, out, k, s, p int, relu bool) *ConvBNReLU {
	c := &ConvBNReLU{BaseModule: NewBase("ConvBNReLU"), UseReLU: relu}
	c.Conv = NewConv2d(in, out, k, s, p, false)
	c.BN = NewBatchNorm2d(out, 1e-5)
	c.RegisterModule("conv", c.Conv)
	c.RegisterModule("bn", c.BN)
	return c
}
func (c *ConvBNReLU) Forward(inputs ...*tensor.Tensor) *tensor.Tensor {
	x := c.BN.Forward(c.Conv.Forward(inputs[0]))
	if c.UseReLU {
		return x.ReLU()
	}
	return x
}
