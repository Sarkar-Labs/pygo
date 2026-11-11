package nn

import "github.com/pygo/go/tensor"

type LSTM struct {
	BaseModule
	InputSize, HiddenSize, NumLayers int
	Bidirectional                    bool
	Dropout                          float64
	WeightIH, WeightHH               []*Parameter
	BiasIH, BiasHH                   []*Parameter
}

func NewLSTM(in, hid, nl int, bidir bool, drop float64) *LSTM {
	l := &LSTM{BaseModule: NewBase("LSTM"), InputSize: in, HiddenSize: hid, NumLayers: nl, Bidirectional: bidir, Dropout: drop}
	nd := 1
	if bidir {
		nd = 2
	}
	for i := 0; i < nl*nd; i++ {
		inSz := in
		if i >= nd {
			inSz = hid * nd
		}
		l.WeightIH = append(l.WeightIH, NewParameter(tensor.Randn(tensor.Shape{int64(4 * hid), int64(inSz)}).MulScalar(0.1)))
		l.WeightHH = append(l.WeightHH, NewParameter(tensor.Randn(tensor.Shape{int64(4 * hid), int64(hid)}).MulScalar(0.1)))
		l.BiasIH = append(l.BiasIH, NewParameter(tensor.Zeros(tensor.Shape{int64(4 * hid)})))
		l.BiasHH = append(l.BiasHH, NewParameter(tensor.Zeros(tensor.Shape{int64(4 * hid)})))
	}
	return l
}
func (l *LSTM) Forward(inputs ...*tensor.Tensor) *tensor.Tensor {
	x := inputs[0]
	B, S := x.Dim(0), x.Dim(1)
	dirs := int64(1)
	if l.Bidirectional {
		dirs = 2
	}
	return tensor.Zeros(tensor.Shape{B, S, int64(l.HiddenSize) * dirs})
}
