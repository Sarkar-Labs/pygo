package data

import (
	"github.com/pygo/go/tensor"
	"math/rand"
)

type Batch struct {
	Inputs, Targets []*tensor.Tensor
	Meta            map[string]interface{}
}
type Dataset interface {
	Len() int
	Get(idx int) *Batch
}

type TensorDataset struct{ Inputs, Targets []*tensor.Tensor }

func NewTensorDataset(inp, tgt []*tensor.Tensor) *TensorDataset {
	return &TensorDataset{Inputs: inp, Targets: tgt}
}
func (d *TensorDataset) Len() int { return len(d.Inputs) }
func (d *TensorDataset) Get(idx int) *Batch {
	return &Batch{Inputs: []*tensor.Tensor{d.Inputs[idx]}, Targets: []*tensor.Tensor{d.Targets[idx]}}
}

type SimpleDataLoader struct {
	dataset   Dataset
	batchSize int
	shuffle   bool
	indices   []int
}

func NewDataLoader(ds Dataset, batchSize int, shuffle bool) *SimpleDataLoader {
	idx := make([]int, ds.Len())
	for i := range idx {
		idx[i] = i
	}
	return &SimpleDataLoader{dataset: ds, batchSize: batchSize, shuffle: shuffle, indices: idx}
}
func (dl *SimpleDataLoader) Len() int { return (dl.dataset.Len() + dl.batchSize - 1) / dl.batchSize }
func (dl *SimpleDataLoader) Iter() chan *Batch {
	ch := make(chan *Batch, 8)
	if dl.shuffle {
		rand.Shuffle(len(dl.indices), func(i, j int) { dl.indices[i], dl.indices[j] = dl.indices[j], dl.indices[i] })
	}
	go func() {
		defer close(ch)
		for i := 0; i < len(dl.indices); i += dl.batchSize {
			end := i + dl.batchSize
			if end > len(dl.indices) {
				end = len(dl.indices)
			}
			var inp, tgt []*tensor.Tensor
			for _, idx := range dl.indices[i:end] {
				b := dl.dataset.Get(idx)
				inp = append(inp, b.Inputs...)
				tgt = append(tgt, b.Targets...)
			}
			ch <- &Batch{Inputs: inp, Targets: tgt}
		}
	}()
	return ch
}
