package nn

import (
	"github.com/pygo/go/tensor"
	"math"
)

type SinusoidalPE struct {
	BaseModule
	MaxLen int
	Drop   *Dropout
	table  *tensor.Tensor
}

func NewSinusoidalPE(maxLen, dim int, dropout float64) *SinusoidalPE {
	pe := &SinusoidalPE{BaseModule: NewBase("SinusoidalPE"), MaxLen: maxLen, Drop: NewDropout(dropout)}
	data := make([]float32, maxLen*dim)
	for pos := 0; pos < maxLen; pos++ {
		for i := 0; i < dim/2; i++ {
			d := math.Pow(10000, float64(2*i)/float64(dim))
			data[pos*dim+2*i] = float32(math.Sin(float64(pos) / d))
			data[pos*dim+2*i+1] = float32(math.Cos(float64(pos) / d))
		}
	}
	pe.table = tensor.FromSlice(data, tensor.Shape{int64(maxLen), int64(dim)})
	return pe
}
func (pe *SinusoidalPE) Forward(inputs ...*tensor.Tensor) *tensor.Tensor {
	x := inputs[0]
	return pe.Drop.Forward(x.Add(pe.table.Slice(0, 0, x.Dim(1), 1).Unsqueeze(0)))
}

type RotaryEmbedding struct {
	BaseModule
	Dim, MaxSeq int
	Theta       float64
	cos, sin    *tensor.Tensor
}

func NewRotaryEmbedding(dim, maxSeq int, theta float64) *RotaryEmbedding {
	r := &RotaryEmbedding{BaseModule: NewBase("RoPE"), Dim: dim, MaxSeq: maxSeq, Theta: theta}
	half := int64(dim / 2)
	inv := make([]float32, half)
	for i := int64(0); i < half; i++ {
		inv[i] = float32(1.0 / math.Pow(theta, float64(2*i)/float64(dim)))
	}
	invT := tensor.FromSlice(inv, tensor.Shape{half})
	t := tensor.Arange(0, float64(maxSeq), 1)
	freqs := t.Unsqueeze(1).MatMul(invT.Unsqueeze(0))
	emb := tensor.Cat([]*tensor.Tensor{freqs, freqs}, -1)
	r.cos = emb.Cos()
	r.sin = emb.Sin()
	return r
}
func (r *RotaryEmbedding) Apply(q, k *tensor.Tensor) (*tensor.Tensor, *tensor.Tensor) {
	S := q.Dim(2)
	cos := r.cos.Slice(0, 0, S, 1).Unsqueeze(0).Unsqueeze(0)
	sin := r.sin.Slice(0, 0, S, 1).Unsqueeze(0).Unsqueeze(0)
	rot := func(x *tensor.Tensor) *tensor.Tensor {
		h := x.Dim(-1) / 2
		return tensor.Cat([]*tensor.Tensor{x.Slice(-1, h, x.Dim(-1), 1).Neg(), x.Slice(-1, 0, h, 1)}, -1)
	}
	return q.Mul(cos).Add(rot(q).Mul(sin)), k.Mul(cos).Add(rot(k).Mul(sin))
}
func (r *RotaryEmbedding) Forward(inputs ...*tensor.Tensor) *tensor.Tensor { panic("Use Apply(q,k)") }
