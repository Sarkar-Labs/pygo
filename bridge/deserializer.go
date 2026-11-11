package bridge

import (
	"context"
	"sync"
)

// GeneratedDeserializerWorker executes callbacks safely and propagates cancellation.
type GeneratedDeserializerWorker struct {
	mu    sync.Mutex
	calls uint64
}

func (w *GeneratedDeserializerWorker) Do(ctx context.Context, fn func(context.Context) error) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	w.mu.Lock()
	w.calls++
	w.mu.Unlock()
	if fn == nil {
		return nil
	}
	return fn(ctx)
}
func (w *GeneratedDeserializerWorker) Calls() uint64 {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.calls
}
