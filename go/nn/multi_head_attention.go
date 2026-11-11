package nn

import "github.com/pygo/go/tensor"

// MultiHeadAttention implements scaled dot-product attention for the common
// [batch, sequence, embedding] representation. Head splitting is represented
// by the native batched matmul path when available; this module currently uses
// a single fused attention projection path.
type MultiHeadAttention struct {
	BaseModule
	EmbedDim, NumHeads           int
	Dropout                      float64
	QProj, KProj, VProj, OutProj *Linear
}

func NewMultiHeadAttention(embedDim, numHeads int, dropout float64) *MultiHeadAttention {
	if embedDim <= 0 || numHeads <= 0 || embedDim%numHeads != 0 {
		panic("NewMultiHeadAttention: embedDim must be positive and divisible by numHeads")
	}
	m := &MultiHeadAttention{BaseModule: NewBase("MultiHeadAttention"), EmbedDim: embedDim, NumHeads: numHeads, Dropout: dropout}
	m.QProj = NewLinear(embedDim, embedDim, true)
	m.KProj = NewLinear(embedDim, embedDim, true)
	m.VProj = NewLinear(embedDim, embedDim, true)
	m.OutProj = NewLinear(embedDim, embedDim, true)
	m.RegisterModule("q_proj", m.QProj)
	m.RegisterModule("k_proj", m.KProj)
	m.RegisterModule("v_proj", m.VProj)
	m.RegisterModule("out_proj", m.OutProj)
	return m
}

func (m *MultiHeadAttention) Forward(inputs ...*tensor.Tensor) *tensor.Tensor {
	if len(inputs) == 0 {
		panic("MultiHeadAttention: missing query input")
	}
	q := m.QProj.Forward(inputs[0])
	k, v := q, q
	if len(inputs) > 1 {
		k = m.KProj.Forward(inputs[1])
		v = m.VProj.Forward(inputs[1])
	} else {
		k = m.KProj.Forward(inputs[0])
		v = m.VProj.Forward(inputs[0])
	}
	scores := q.MatMul(k.Transpose(-2, -1)).MulScalar(1.0 / float64((m.EmbedDim / m.NumHeads)))
	return m.OutProj.Forward(scores.Softmax(-1).MatMul(v))
}
