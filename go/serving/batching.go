package serving

import (
	"context"
	"github.com/pygo/go/tensor"
	"sync"
	"time"
)

type BatchRequest struct {
	ID       string
	Input    *tensor.Tensor
	ResultCh chan *BatchResult
	Ctx      context.Context
}
type BatchResult struct {
	ID     string
	Output *tensor.Tensor
	Error  error
}
type DynamicBatcher struct {
	mu        sync.Mutex
	cfg       ServerConfig
	queue     []*BatchRequest
	triggerCh chan struct{}
	processFn func([]*tensor.Tensor) []*tensor.Tensor
}

func NewDynamicBatcher(cfg ServerConfig, fn func([]*tensor.Tensor) []*tensor.Tensor) *DynamicBatcher {
	b := &DynamicBatcher{cfg: cfg, processFn: fn, triggerCh: make(chan struct{}, 1)}
	go b.loop()
	return b
}
func (b *DynamicBatcher) Submit(req *BatchRequest) {
	b.mu.Lock()
	b.queue = append(b.queue, req)
	b.mu.Unlock()
	select {
	case b.triggerCh <- struct{}{}:
	default:
	}
}
func (b *DynamicBatcher) loop() {
	ticker := time.NewTicker(b.cfg.BatchTimeout)
	defer ticker.Stop()
	for {
		select {
		case <-ticker.C:
			b.flush()
		case <-b.triggerCh:
			b.mu.Lock()
			n := len(b.queue)
			b.mu.Unlock()
			if n >= b.cfg.MaxBatchSize {
				b.flush()
			}
		}
	}
}
func (b *DynamicBatcher) flush() {
	b.mu.Lock()
	batch := b.queue
	b.queue = nil
	b.mu.Unlock()
	if len(batch) == 0 {
		return
	}
	inputs := make([]*tensor.Tensor, len(batch))
	for i, r := range batch {
		inputs[i] = r.Input
	}
	outputs := b.processFn(inputs)
	for i, req := range batch {
		var out *tensor.Tensor
		if i < len(outputs) {
			out = outputs[i]
		}
		select {
		case req.ResultCh <- &BatchResult{ID: req.ID, Output: out}:
		default:
		}
	}
}
