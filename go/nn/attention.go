package nn

import "github.com/pygo/go/tensor"

type GroupedQueryAttention struct {
	BaseModule
	EmbedDim, NumQHeads, NumKVHeads, HeadDim, NRep int
	Scale                                          float64
	QProj, KProj, VProj, OProj                     *Linear
}

func NewGQA(d, nQ, nKV int, drop float64) *GroupedQueryAttention {
	a := &GroupedQueryAttention{BaseModule: NewBase("GQA"), EmbedDim: d, NumQHeads: nQ, NumKVHeads: nKV, HeadDim: d / nQ, NRep: nQ / nKV, Scale: 1.0 / float64(d/nQ)}
	a.QProj = NewLinear(d, d, false)
	a.KProj = NewLinear(d, (d/nQ)*nKV, false)
	a.VProj = NewLinear(d, (d/nQ)*nKV, false)
	a.OProj = NewLinear(d, d, false)
	a.RegisterModule("q_proj", a.QProj)
	a.RegisterModule("k_proj", a.KProj)
	a.RegisterModule("v_proj", a.VProj)
	a.RegisterModule("o_proj", a.OProj)
	return a
}
func (a *GroupedQueryAttention) Forward(inputs ...*tensor.Tensor) *tensor.Tensor {
	x := inputs[0]
	B, S, D := x.Dim(0), x.Dim(1), x.Dim(2)
	nH, nKV, Hd := int64(a.NumQHeads), int64(a.NumKVHeads), int64(a.HeadDim)
	q := a.QProj.Forward(x).Reshape(tensor.Shape{B, S, nH, Hd}).Permute([]int{0, 2, 1, 3})
	k := a.KProj.Forward(x).Reshape(tensor.Shape{B, S, nKV, Hd}).Permute([]int{0, 2, 1, 3})
	v := a.VProj.Forward(x).Reshape(tensor.Shape{B, S, nKV, Hd}).Permute([]int{0, 2, 1, 3})
	if a.NRep > 1 {
		k = k.Expand(B, nH, S, Hd)
		v = v.Expand(B, nH, S, Hd)
	}
	out := q.MatMul(k.Transpose(-2, -1)).MulScalar(a.Scale).Softmax(-1).MatMul(v).Permute([]int{0, 2, 1, 3}).Reshape(tensor.Shape{B, S, D})
	return a.OProj.Forward(out)
}
