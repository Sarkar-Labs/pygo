package training

import (
	"github.com/pygo/go/nn"
	"github.com/pygo/go/tensor"
)

type ColumnParallelLinear struct {
	nn.BaseModule
	base            *nn.Linear
	rank, worldSize int
}

func NewColumnParallelLinear(in, out, rank, worldSize int) *ColumnParallelLinear {
	return &ColumnParallelLinear{BaseModule: nn.NewBase("ColParallel"), base: nn.NewLinear(in, (out+worldSize-1)/worldSize, true), rank: rank, worldSize: worldSize}
}
func (l *ColumnParallelLinear) Forward(inputs ...*tensor.Tensor) *tensor.Tensor {
	return l.base.Forward(inputs[0])
}

type RowParallelLinear struct {
	nn.BaseModule
	base            *nn.Linear
	rank, worldSize int
}

func NewRowParallelLinear(in, out, rank, worldSize int) *RowParallelLinear {
	li := (in + worldSize - 1) / worldSize
	return &RowParallelLinear{BaseModule: nn.NewBase("RowParallel"), base: nn.NewLinear(li, out, false), rank: rank, worldSize: worldSize}
}
func (l *RowParallelLinear) Forward(inputs ...*tensor.Tensor) *tensor.Tensor {
	li := inputs[0].Dim(-1) / int64(l.worldSize)
	s := int64(l.rank) * li
	return l.base.Forward(inputs[0].Slice(-1, s, s+li, 1))
}
