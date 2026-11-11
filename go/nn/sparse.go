package nn

import (
	"fmt"
	"github.com/pygo/go/tensor"
)

type SparseLinear struct {
	BaseModule
	InFeatures, OutFeatures int
	Sparsity                float64
	Weight                  *Parameter
	Mask                    *tensor.Tensor
	Bias                    *Parameter
}

func NewSparseLinear(in, out int, sparsity float64, bias bool) *SparseLinear {
	sl := &SparseLinear{BaseModule: NewBase(fmt.Sprintf("SparseLinear(%d->%d)", in, out)), InFeatures: in, OutFeatures: out, Sparsity: sparsity}
	sl.Weight = NewParameter(tensor.Randn(tensor.Shape{int64(out), int64(in)}).MulScalar(0.01))
	sl.Mask = tensor.Ones(tensor.Shape{int64(out), int64(in)})
	sl.RegisterParameter("weight", sl.Weight)
	if bias {
		sl.Bias = NewParameter(tensor.Zeros(tensor.Shape{int64(out)}))
		sl.RegisterParameter("bias", sl.Bias)
	}
	return sl
}
func (sl *SparseLinear) Forward(inputs ...*tensor.Tensor) *tensor.Tensor {
	w := sl.Weight.Tensor.Mul(sl.Mask)
	out := inputs[0].MatMul(w.Transpose(-2, -1))
	if sl.Bias != nil {
		out = out.Add(sl.Bias.Tensor)
	}
	return out
}
