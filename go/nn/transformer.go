package nn

import (
	"fmt"
	"github.com/pygo/go/tensor"
)

type TransformerEncoderLayer struct {
	BaseModule
	SelfAttn     *MultiHeadAttention
	FFN          *Sequential
	Norm1, Norm2 *LayerNorm
	Drop         *Dropout
}

func NewTransformerEncoderLayer(d, h, ff int, drop float64) *TransformerEncoderLayer {
	t := &TransformerEncoderLayer{BaseModule: NewBase(fmt.Sprintf("TEncLayer(d=%d,h=%d)", d, h))}
	t.SelfAttn = NewMultiHeadAttention(d, h, drop)
	t.FFN = NewSequential(NewLinear(d, ff, true), NewGELU(), NewDropout(drop), NewLinear(ff, d, true))
	t.Norm1 = NewLayerNorm([]int{d}, 1e-5)
	t.Norm2 = NewLayerNorm([]int{d}, 1e-5)
	t.Drop = NewDropout(drop)
	t.RegisterModule("self_attn", t.SelfAttn)
	t.RegisterModule("ffn", t.FFN)
	return t
}
func (t *TransformerEncoderLayer) Forward(inputs ...*tensor.Tensor) *tensor.Tensor {
	x := inputs[0]
	x = x.Add(t.Drop.Forward(t.SelfAttn.Forward(t.Norm1.Forward(x))))
	return x.Add(t.Drop.Forward(t.FFN.Forward(t.Norm2.Forward(x))))
}

type TransformerDecoderLayer struct {
	BaseModule
	SelfAttn, CrossAttn *MultiHeadAttention
	FFN                 *Sequential
	Norm1, Norm2, Norm3 *LayerNorm
	Drop                *Dropout
}

func NewTransformerDecoderLayer(d, h, ff int, drop float64) *TransformerDecoderLayer {
	t := &TransformerDecoderLayer{BaseModule: NewBase(fmt.Sprintf("TDecLayer(d=%d)", d))}
	t.SelfAttn = NewMultiHeadAttention(d, h, drop)
	t.CrossAttn = NewMultiHeadAttention(d, h, drop)
	t.FFN = NewSequential(NewLinear(d, ff, true), NewGELU(), NewLinear(ff, d, true))
	t.Norm1 = NewLayerNorm([]int{d}, 1e-5)
	t.Norm2 = NewLayerNorm([]int{d}, 1e-5)
	t.Norm3 = NewLayerNorm([]int{d}, 1e-5)
	t.Drop = NewDropout(drop)
	return t
}
func (t *TransformerDecoderLayer) Forward(inputs ...*tensor.Tensor) *tensor.Tensor {
	x, mem := inputs[0], inputs[1]
	x = x.Add(t.Drop.Forward(t.SelfAttn.Forward(t.Norm1.Forward(x))))
	x = x.Add(t.Drop.Forward(t.CrossAttn.Forward(t.Norm2.Forward(x), mem)))
	return x.Add(t.Drop.Forward(t.FFN.Forward(t.Norm3.Forward(x))))
}
