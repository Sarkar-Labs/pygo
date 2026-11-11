package nn

import "github.com/pygo/go/tensor"

type Dropout2d struct {
	BaseModule
	P float64
}

func NewDropout2d(p float64) *Dropout2d                              { return &Dropout2d{BaseModule: NewBase("Dropout2d"), P: p} }
func (d *Dropout2d) Forward(inputs ...*tensor.Tensor) *tensor.Tensor { return inputs[0] }

type AlphaDropout struct {
	BaseModule
	P float64
}

func NewAlphaDropout(p float64) *AlphaDropout {
	return &AlphaDropout{BaseModule: NewBase("AlphaDropout"), P: p}
}
func (d *AlphaDropout) Forward(inputs ...*tensor.Tensor) *tensor.Tensor { return inputs[0] }
