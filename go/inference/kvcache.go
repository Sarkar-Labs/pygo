package inference

import (
	"github.com/pygo/go/tensor"
	"sync"
)

type KVCache struct {
	mu                 sync.RWMutex
	Keys, Values       []*tensor.Tensor
	SeenTokens, MaxLen int64
	NumLayers          int
}

func NewKVCache(numLayers int, maxLen int64) *KVCache {
	return &KVCache{Keys: make([]*tensor.Tensor, numLayers), Values: make([]*tensor.Tensor, numLayers), NumLayers: numLayers, MaxLen: maxLen}
}
func (kv *KVCache) Update(layer int, k, v *tensor.Tensor) {
	kv.mu.Lock()
	defer kv.mu.Unlock()
	if kv.Keys[layer] == nil {
		kv.Keys[layer] = k
		kv.Values[layer] = v
	} else {
		kv.Keys[layer] = tensor.Cat([]*tensor.Tensor{kv.Keys[layer], k}, 2)
		kv.Values[layer] = tensor.Cat([]*tensor.Tensor{kv.Values[layer], v}, 2)
		if kv.Keys[layer].Dim(2) > kv.MaxLen {
			kv.Keys[layer] = kv.Keys[layer].Slice(2, -kv.MaxLen, kv.Keys[layer].Dim(2), 1)
			kv.Values[layer] = kv.Values[layer].Slice(2, -kv.MaxLen, kv.Values[layer].Dim(2), 1)
		}
	}
	kv.SeenTokens += k.Dim(2)
}
func (kv *KVCache) Get(layer int) (*tensor.Tensor, *tensor.Tensor) {
	kv.mu.RLock()
	defer kv.mu.RUnlock()
	return kv.Keys[layer], kv.Values[layer]
}
func (kv *KVCache) Reset() {
	kv.mu.Lock()
	defer kv.mu.Unlock()
	for i := range kv.Keys {
		kv.Keys[i] = nil
		kv.Values[i] = nil
	}
	kv.SeenTokens = 0
}
func (kv *KVCache) Empty() bool { kv.mu.RLock(); defer kv.mu.RUnlock(); return kv.SeenTokens == 0 }
