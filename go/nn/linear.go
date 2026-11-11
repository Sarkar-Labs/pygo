package nn

import "github.com/pygo/go/tensor"

type Bilinear struct {
	BaseModule
	In1, In2, Out int
	Weight        *Parameter
	Bias          *Parameter
}

func NewBilinear(in1, in2, out int, bias bool) *Bilinear {
	b := &Bilinear{BaseModule: NewBase("Bilinear"), In1: in1, In2: in2, Out: out}
	b.Weight = NewParameter(tensor.Randn(tensor.Shape{int64(out), int64(in1), int64(in2)}).MulScalar(0.01))
	b.RegisterParameter("weight", b.Weight)
	if bias {
		b.Bias = NewParameter(tensor.Zeros(tensor.Shape{int64(out)}))
		b.RegisterParameter("bias", b.Bias)
	}
	return b
}
func (b *Bilinear) Forward(inputs ...*tensor.Tensor) *tensor.Tensor {
	out := inputs[0].MatMul(b.Weight.Tensor.Reshape(tensor.Shape{int64(b.Out), -1}))
	if b.Bias != nil {
		out = out.Add(b.Bias.Tensor)
	}
	return out
}

type Identity struct{ BaseModule }

func NewIdentity() *Identity                                        { return &Identity{BaseModule: NewBase("Identity")} }
func (i *Identity) Forward(inputs ...*tensor.Tensor) *tensor.Tensor { return inputs[0] }

type Flatten struct {
	BaseModule
	StartDim, EndDim int
}

func NewFlatten(s, e int) *Flatten {
	return &Flatten{BaseModule: NewBase("Flatten"), StartDim: s, EndDim: e}
}
func (f *Flatten) Forward(inputs ...*tensor.Tensor) *tensor.Tensor {
	return inputs[0].Flatten(f.StartDim, f.EndDim)
}
