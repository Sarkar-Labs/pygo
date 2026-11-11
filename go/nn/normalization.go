package nn

import "github.com/pygo/go/tensor"

type RMSNorm struct {
	BaseModule
	Dim    int
	Eps    float64
	Weight *Parameter
}

func NewRMSNorm(dim int, eps float64) *RMSNorm {
	r := &RMSNorm{BaseModule: NewBase("RMSNorm"), Dim: dim, Eps: eps}
	r.Weight = NewParameter(tensor.Ones(tensor.Shape{int64(dim)}))
	r.RegisterParameter("weight", r.Weight)
	return r
}
func (r *RMSNorm) Forward(inputs ...*tensor.Tensor) *tensor.Tensor {
	x := inputs[0]
	return x.Div(x.Pow(2).Mean(-1, true).AddScalar(r.Eps).Sqrt()).Mul(r.Weight.Tensor)
}

type GroupNorm struct {
	BaseModule
	NumGroups, NumChannels int
	Eps                    float64
	Weight, Bias           *Parameter
}

func NewGroupNorm(g, c int, eps float64) *GroupNorm {
	gn := &GroupNorm{BaseModule: NewBase("GroupNorm"), NumGroups: g, NumChannels: c, Eps: eps}
	gn.Weight = NewParameter(tensor.Ones(tensor.Shape{int64(c)}))
	gn.RegisterParameter("weight", gn.Weight)
	gn.Bias = NewParameter(tensor.Zeros(tensor.Shape{int64(c)}))
	gn.RegisterParameter("bias", gn.Bias)
	return gn
}
func (g *GroupNorm) Forward(inputs ...*tensor.Tensor) *tensor.Tensor { return inputs[0] }
