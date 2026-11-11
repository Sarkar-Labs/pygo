package nn

import (
	"fmt"
	"github.com/pygo/go/tensor"
)

type Bottleneck struct {
	BaseModule
	Conv1, Conv2, Conv3 *Conv2d
	BN1, BN2, BN3       *BatchNorm2d
	Downsample          *Sequential
}

func NewBottleneck(inCh, midCh, stride int, ds *Sequential) *Bottleneck {
	b := &Bottleneck{BaseModule: NewBase(fmt.Sprintf("Bottleneck(%d,%d)", inCh, midCh))}
	b.Conv1 = NewConv2d(inCh, midCh, 1, 1, 0, false)
	b.BN1 = NewBatchNorm2d(midCh, 1e-5)
	b.Conv2 = NewConv2d(midCh, midCh, 3, stride, 1, false)
	b.BN2 = NewBatchNorm2d(midCh, 1e-5)
	b.Conv3 = NewConv2d(midCh, midCh*4, 1, 1, 0, false)
	b.BN3 = NewBatchNorm2d(midCh*4, 1e-5)
	b.Downsample = ds
	b.RegisterModule("conv1", b.Conv1)
	b.RegisterModule("conv3", b.Conv3)
	return b
}
func (b *Bottleneck) Forward(inputs ...*tensor.Tensor) *tensor.Tensor {
	x := inputs[0]
	out := b.BN1.Forward(b.Conv1.Forward(x)).ReLU()
	out = b.BN2.Forward(b.Conv2.Forward(out)).ReLU()
	out = b.BN3.Forward(b.Conv3.Forward(out))
	id := x
	if b.Downsample != nil {
		id = b.Downsample.Forward(x)
	}
	return out.Add(id).ReLU()
}

type BasicBlock struct {
	BaseModule
	Conv1, Conv2 *Conv2d
	BN1, BN2     *BatchNorm2d
	Downsample   *Sequential
}

func NewBasicBlock(inCh, outCh, stride int, ds *Sequential) *BasicBlock {
	b := &BasicBlock{BaseModule: NewBase(fmt.Sprintf("BasicBlock(%d)", outCh))}
	b.Conv1 = NewConv2d(inCh, outCh, 3, stride, 1, false)
	b.BN1 = NewBatchNorm2d(outCh, 1e-5)
	b.Conv2 = NewConv2d(outCh, outCh, 3, 1, 1, false)
	b.BN2 = NewBatchNorm2d(outCh, 1e-5)
	b.Downsample = ds
	return b
}
func (b *BasicBlock) Forward(inputs ...*tensor.Tensor) *tensor.Tensor {
	x := inputs[0]
	out := b.BN1.Forward(b.Conv1.Forward(x)).ReLU()
	out = b.BN2.Forward(b.Conv2.Forward(out))
	id := x
	if b.Downsample != nil {
		id = b.Downsample.Forward(x)
	}
	return out.Add(id).ReLU()
}
func MakeResLayer(inCh, midCh, n, stride int) *Sequential {
	var ds *Sequential
	if stride != 1 || inCh != midCh*4 {
		ds = NewSequential(NewConv2d(inCh, midCh*4, 1, stride, 0, false), NewBatchNorm2d(midCh*4, 1e-5))
	}
	blocks := []Module{NewBottleneck(inCh, midCh, stride, ds)}
	for i := 1; i < n; i++ {
		blocks = append(blocks, NewBottleneck(midCh*4, midCh, 1, nil))
	}
	return NewSequential(blocks...)
}
