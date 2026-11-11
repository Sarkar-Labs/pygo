package nn

import (
	"fmt"
	"github.com/pygo/go/tensor"
)

type LoRALinear struct {
	BaseModule
	InFeatures, OutFeatures, Rank int
	Scaling                       float64
	Weight, LoRAA, LoRAB          *Parameter
	Bias                          *Parameter
}

func NewLoRALinear(in, out, r int, alpha float64, bias bool) *LoRALinear {
	l := &LoRALinear{BaseModule: NewBase(fmt.Sprintf("LoRA(%d->%d,r=%d)", in, out, r)), InFeatures: in, OutFeatures: out, Rank: r, Scaling: alpha / float64(r)}
	l.Weight = NewParameter(tensor.Randn(tensor.Shape{int64(out), int64(in)}).MulScalar(0.02))
	l.Weight.SetRequiresGrad(false)
	l.LoRAA = NewParameter(tensor.Randn(tensor.Shape{int64(r), int64(in)}).MulScalar(0.01))
	l.LoRAB = NewParameter(tensor.Zeros(tensor.Shape{int64(out), int64(r)}))
	l.RegisterParameter("weight", l.Weight)
	l.RegisterParameter("lora_A", l.LoRAA)
	l.RegisterParameter("lora_B", l.LoRAB)
	if bias {
		l.Bias = NewParameter(tensor.Zeros(tensor.Shape{int64(out)}))
		l.RegisterParameter("bias", l.Bias)
	}
	return l
}
func (l *LoRALinear) Forward(inputs ...*tensor.Tensor) *tensor.Tensor {
	x := inputs[0]
	base := x.MatMul(l.Weight.Tensor.Transpose(-2, -1))
	lora := x.MatMul(l.LoRAA.Tensor.Transpose(-2, -1)).MatMul(l.LoRAB.Tensor.Transpose(-2, -1)).MulScalar(l.Scaling)
	out := base.Add(lora)
	if l.Bias != nil {
		out = out.Add(l.Bias.Tensor)
	}
	return out
}
func (l *LoRALinear) MergeWeights() *Linear {
	m := NewLinear(l.InFeatures, l.OutFeatures, l.Bias != nil)
	m.Weight = NewParameter(l.Weight.Tensor.Add(l.LoRAB.Tensor.MatMul(l.LoRAA.Tensor).MulScalar(l.Scaling)))
	return m
}
func (l *LoRALinear) TrainableParams() []*Parameter { return []*Parameter{l.LoRAA, l.LoRAB} }
