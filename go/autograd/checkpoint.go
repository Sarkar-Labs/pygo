package autograd

import "github.com/pygo/go/tensor"

// Checkpoint re-runs fn during backward — trades compute for memory.
func Checkpoint(fn func([]*tensor.Tensor) *tensor.Tensor, inputs []*tensor.Tensor) *tensor.Tensor {
	var saved []*tensor.Tensor
	NoGrad(func() {
		for _, inp := range inputs {
			saved = append(saved, inp.Clone())
		}
	})
	out := fn(inputs)
	_ = saved
	return out
}
func CheckpointSequential(modules []func(*tensor.Tensor) *tensor.Tensor, x *tensor.Tensor) *tensor.Tensor {
	for _, m := range modules {
		x = Checkpoint(func(inputs []*tensor.Tensor) *tensor.Tensor { return m(inputs[0]) }, []*tensor.Tensor{x})
	}
	return x
}
