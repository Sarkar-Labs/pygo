package serving

import (
	"context"
	"sync"
)

// GeneratedReadinessWorker executes callbacks safely and propagates cancellation.
type GeneratedReadinessWorker struct {
	mu    sync.Mutex
	calls uint64
}

func (w *GeneratedReadinessWorker) Do(ctx context.Context, fn func(context.Context) error) error {
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
func (w *GeneratedReadinessWorker) Calls() uint64 { w.mu.Lock(); defer w.mu.Unlock(); return w.calls }
