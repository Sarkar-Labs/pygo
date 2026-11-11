package nn

import "github.com/pygo/go/tensor"

type ReLU struct {
	BaseModule
	Inplace bool
}
type GELU struct{ BaseModule }
type SiLU struct{ BaseModule }
type Tanh struct{ BaseModule }
type Sigmoid struct{ BaseModule }
type Softmax struct {
	BaseModule
	Dim int
}
type LeakyReLU struct {
	BaseModule
	NegSlope float64
}
type Mish struct{ BaseModule }
type HardSwish struct{ BaseModule }
type SELU struct{ BaseModule }

func NewReLU(inplace bool) *ReLU  { return &ReLU{BaseModule: NewBase("ReLU"), Inplace: inplace} }
func NewGELU() *GELU              { return &GELU{BaseModule: NewBase("GELU")} }
func NewSiLU() *SiLU              { return &SiLU{BaseModule: NewBase("SiLU")} }
func NewTanh() *Tanh              { return &Tanh{BaseModule: NewBase("Tanh")} }
func NewSigmoid() *Sigmoid        { return &Sigmoid{BaseModule: NewBase("Sigmoid")} }
func NewSoftmax(dim int) *Softmax { return &Softmax{BaseModule: NewBase("Softmax"), Dim: dim} }
func NewLeakyReLU(s float64) *LeakyReLU {
	return &LeakyReLU{BaseModule: NewBase("LeakyReLU"), NegSlope: s}
}
func NewMish() *Mish                                                 { return &Mish{BaseModule: NewBase("Mish")} }
func NewHardSwish() *HardSwish                                       { return &HardSwish{BaseModule: NewBase("HardSwish")} }
func NewSELU() *SELU                                                 { return &SELU{BaseModule: NewBase("SELU")} }
func (r *ReLU) Forward(inputs ...*tensor.Tensor) *tensor.Tensor      { return inputs[0].ReLU() }
func (g *GELU) Forward(inputs ...*tensor.Tensor) *tensor.Tensor      { return inputs[0].GELU() }
func (s *SiLU) Forward(inputs ...*tensor.Tensor) *tensor.Tensor      { return inputs[0].SiLU() }
func (t *Tanh) Forward(inputs ...*tensor.Tensor) *tensor.Tensor      { return inputs[0].Tanh() }
func (s *Sigmoid) Forward(inputs ...*tensor.Tensor) *tensor.Tensor   { return inputs[0].Sigmoid() }
func (s *Softmax) Forward(inputs ...*tensor.Tensor) *tensor.Tensor   { return inputs[0].Softmax(s.Dim) }
func (l *LeakyReLU) Forward(inputs ...*tensor.Tensor) *tensor.Tensor { return inputs[0] }
func (m *Mish) Forward(inputs ...*tensor.Tensor) *tensor.Tensor {
	return inputs[0].Mul(inputs[0].Tanh())
}
func (h *HardSwish) Forward(inputs ...*tensor.Tensor) *tensor.Tensor {
	return inputs[0].Clamp(0, 6).Mul(inputs[0]).MulScalar(1.0 / 6.0)
}
func (s *SELU) Forward(inputs ...*tensor.Tensor) *tensor.Tensor { return inputs[0] }
