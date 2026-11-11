package nn

import (
	"fmt"
	"github.com/pygo/go/tensor"
)

type BatchNorm2d struct {
	BaseModule
	NumFeatures             int
	Eps, Momentum           float64
	Weight, Bias            *Parameter
	RunningMean, RunningVar *tensor.Tensor
}

func NewBatchNorm2d(nf int, eps float64) *BatchNorm2d {
	bn := &BatchNorm2d{BaseModule: NewBase(fmt.Sprintf("BN2d(%d)", nf)), NumFeatures: nf, Eps: eps, Momentum: 0.1}
	bn.Weight = NewParameter(tensor.Ones(tensor.Shape{int64(nf)}))
	bn.RegisterParameter("weight", bn.Weight)
	bn.Bias = NewParameter(tensor.Zeros(tensor.Shape{int64(nf)}))
	bn.RegisterParameter("bias", bn.Bias)
	bn.RunningMean = tensor.Zeros(tensor.Shape{int64(nf)})
	bn.RunningVar = tensor.Ones(tensor.Shape{int64(nf)})
	return bn
}
func (bn *BatchNorm2d) Forward(inputs ...*tensor.Tensor) *tensor.Tensor {
	x := inputs[0]
	nf := int64(bn.NumFeatures)
	if !bn.IsTraining() {
		m := bn.RunningMean.Reshape(tensor.Shape{1, nf, 1, 1})
		s := bn.RunningVar.AddScalar(bn.Eps).Sqrt().Reshape(tensor.Shape{1, nf, 1, 1})
		return x.Sub(m).Div(s).Mul(bn.Weight.Tensor.Reshape(tensor.Shape{1, nf, 1, 1})).Add(bn.Bias.Tensor.Reshape(tensor.Shape{1, nf, 1, 1}))
	}
	m := x.Mean(0, true).Mean(2, true).Mean(3, true)
	v := x.Sub(m).Pow(2).Mean(0, true).Mean(2, true).Mean(3, true)
	xn := x.Sub(m).Div(v.AddScalar(bn.Eps).Sqrt())
	return xn.Mul(bn.Weight.Tensor.Reshape(tensor.Shape{1, nf, 1, 1})).Add(bn.Bias.Tensor.Reshape(tensor.Shape{1, nf, 1, 1}))
}
