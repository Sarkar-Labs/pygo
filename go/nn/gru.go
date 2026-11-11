package nn

import "github.com/pygo/go/tensor"

type GRU struct {
	BaseModule
	InputSize, HiddenSize, NumLayers int
	Bidirectional                    bool
	Dropout                          float64
	WeightIH, WeightHH               []*Parameter
	BiasIH, BiasHH                   []*Parameter
}

func NewGRU(in, hid, nl int, bidir bool, drop float64) *GRU {
	g := &GRU{BaseModule: NewBase("GRU"), InputSize: in, HiddenSize: hid, NumLayers: nl, Bidirectional: bidir, Dropout: drop}
	nd := 1
	if bidir {
		nd = 2
	}
	for l := 0; l < nl*nd; l++ {
		inSz := in
		if l >= nd {
			inSz = hid * nd
		}
		g.WeightIH = append(g.WeightIH, NewParameter(tensor.Randn(tensor.Shape{int64(3 * hid), int64(inSz)}).MulScalar(0.1)))
		g.WeightHH = append(g.WeightHH, NewParameter(tensor.Randn(tensor.Shape{int64(3 * hid), int64(hid)}).MulScalar(0.1)))
		g.BiasIH = append(g.BiasIH, NewParameter(tensor.Zeros(tensor.Shape{int64(3 * hid)})))
		g.BiasHH = append(g.BiasHH, NewParameter(tensor.Zeros(tensor.Shape{int64(3 * hid)})))
	}
	return g
}
func (g *GRU) Forward(inputs ...*tensor.Tensor) *tensor.Tensor {
	x := inputs[0]
	B, S := x.Dim(0), x.Dim(1)
	dirs := int64(1)
	if g.Bidirectional {
		dirs = 2
	}
	return tensor.Zeros(tensor.Shape{B, S, int64(g.HiddenSize) * dirs})
}
