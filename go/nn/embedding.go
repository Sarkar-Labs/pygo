package nn

import "github.com/pygo/go/tensor"

type EmbeddingBag struct {
	BaseModule
	NumEmbeddings, EmbeddingDim int
	Mode                        string
	Weight                      *Parameter
}

func NewEmbeddingBag(num, dim int, mode string) *EmbeddingBag {
	e := &EmbeddingBag{BaseModule: NewBase("EmbeddingBag"), NumEmbeddings: num, EmbeddingDim: dim, Mode: mode}
	e.Weight = NewParameter(tensor.Randn(tensor.Shape{int64(num), int64(dim)}).MulScalar(0.02))
	e.RegisterParameter("weight", e.Weight)
	return e
}
func (e *EmbeddingBag) Forward(inputs ...*tensor.Tensor) *tensor.Tensor {
	return e.Weight.Tensor.Mean(-1, false)
}
