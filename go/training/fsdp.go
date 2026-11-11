package training

import (
	"fmt"
	"github.com/pygo/go/nn"
	"github.com/pygo/go/tensor"
	"sync"
)

type FSDPConfig struct {
	Strategy                   int
	CPUOffload, MixedPrecision bool
	MinNumParams               int64
}

func DefaultFSDPConfig() FSDPConfig { return FSDPConfig{MixedPrecision: true, MinNumParams: 1_000_000} }

type FSDPWrapper struct {
	mu              sync.Mutex
	model           nn.Module
	cfg             FSDPConfig
	rank, worldSize int
	shards          map[string]*tensor.Tensor
}

func WrapFSDP(model nn.Module, cfg FSDPConfig, rank, worldSize int) *FSDPWrapper {
	f := &FSDPWrapper{model: model, cfg: cfg, rank: rank, worldSize: worldSize, shards: make(map[string]*tensor.Tensor)}
	f.mu.Lock()
	defer f.mu.Unlock()
	for name, p := range model.NamedParameters() {
		n := p.NumEl()
		sz := (n + int64(worldSize) - 1) / int64(worldSize)
		s := int64(rank) * sz
		e := s + sz
		if e > n {
			e = n
		}
		f.shards[name] = p.Slice(0, s, e, 1)
	}
	fmt.Printf("FSDP rank=%d: %d shards\n", rank, len(f.shards))
	return f
}
func (f *FSDPWrapper) Forward(inputs ...*tensor.Tensor) *tensor.Tensor {
	return f.model.Forward(inputs...)
}
func (f *FSDPWrapper) Parameters() []*nn.Parameter { return f.model.Parameters() }
func (f *FSDPWrapper) ZeroGrad()                   { f.model.ZeroGrad() }
func (f *FSDPWrapper) Train(mode bool)             { f.model.Train(mode) }
